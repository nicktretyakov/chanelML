use std::sync::Arc;
use std::time::{Duration, Instant};
use anyhow::{anyhow, Result};
use crossbeam_channel::{bounded, select, unbounded};
use tch::Tensor;
use tonic::{transport::Server, Request, Response, Status, Streaming};
use tract_onnx::prelude::*;
use rand::Rng;
use tokio_stream::StreamExt;

// Protobuf структуры (сгенерированы prost)
pub mod ml {
    include!(concat!(env!("OUT_DIR"), "/ml.rs"));
}

// Type alias for model
type OnnxModel = SimplePlan<TypedFact, Box<dyn TypedOp>, tract_onnx::prelude::Graph<TypedFact, Box<dyn TypedOp>>>;

// ONNX Runtime сервис
struct OnnxRuntime {
    model: Arc<OnnxModel>,
    task_sender: crossbeam_channel::Sender<(Tensor, crossbeam_channel::Sender<Result<Tensor>>)>,
}

impl OnnxRuntime {
    fn new(model_path: &str) -> Result<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(model_path)?
            .into_optimized()?
            .into_runnable()?;
        
        let (task_sender, task_receiver) = unbounded::<(Tensor, crossbeam_channel::Sender<Result<Tensor>>)>();
        let model_ref = Arc::new(model);
        let model_ref_worker = model_ref.clone();
        
        // GPU Worker
        std::thread::spawn(move || {
            while let Ok((input, result_sender)) = task_receiver.recv() {
                let model = model_ref_worker.clone();
                let result = Self::process_input(model, input);
                let _ = result_sender.send(result);
            }
        });
        
        Ok(Self {
            model: model_ref,
            task_sender,
        })
    }

    fn process_input(model: Arc<OnnxModel>, input: Tensor) -> Result<Tensor> {
        // Конвертация tch::Tensor в tract
        let shape: Vec<usize> = input.size().iter().map(|&d| d as usize).collect();
        let values: Vec<f32> = Vec::<f32>::try_from(input)?;
        let tract_array = tract_ndarray::Array::from_shape_vec(&*shape, values)?;
        let tract_tensor = tract_array.into_tvalue();
        
        // Выполнение инференса
        let result = model.run(tvec!(tract_tensor))?;
        let output = result[0].to_array_view::<f32>()?;
        
        // Конвертация обратно в tch::Tensor
        let output_shape: Vec<i64> = output.shape().iter().map(|&d| d as i64).collect();
        Ok(Tensor::f_from_slice(output.as_slice().unwrap())?.reshape(&output_shape))
    }
}

// gRPC сервер для распределённой обработки
#[tonic::async_trait]
impl ml::inference_service_server::InferenceService for OnnxRuntime {
    async fn predict(
        &self,
        request: Request<Streaming<ml::TensorRequest>>,
    ) -> Result<Response<ml::TensorResponse>, Status> {
        let (result_sender, result_receiver) = bounded(10);
        let mut stream = request.into_inner();
        let start_time = Instant::now();

        // Поток обработки запросов
        tokio::spawn(async move {
            while let Some(req) = stream.next().await {
                match req {
                    Ok(tensor_req) => {
                        match Self::deserialize_tensor_request(&tensor_req) {
                            Ok(tensor) => {
                                if let Err(e) = result_sender.send(tensor) {
                                    eprintln!("Send error: {}", e);
                                    break;
                                }
                            }
                            Err(e) => eprintln!("Deserialization error: {}", e),
                        }
                    }
                    Err(e) => {
                        eprintln!("Stream error: {}", e);
                        break;
                    }
                }
            }
        });

        // Сбор результатов
        let mut outputs = Vec::new();
        while let Ok(tensor) = result_receiver.recv() {
            if let Ok(result) = self.predict_tensor(tensor) {
                outputs.push(result);
            }
        }

        let response = ml::TensorResponse {
            data: outputs,
            latency_ms: start_time.elapsed().as_millis() as u32,
        };

        Ok(Response::new(response))
    }
}

impl OnnxRuntime {
    fn predict_tensor(&self, input: Tensor) -> Result<ml::TensorProto> {
        let (sender, receiver) = bounded(1);
        match self.task_sender.send((input, sender)) {
            Ok(_) => {},
            Err(_) => return Err(anyhow!("Failed to send tensor to worker")),
        }
        
        select! {
            recv(receiver) -> result => {
                match result {
                    Ok(tensor_result) => {
                        match tensor_result {
                            Ok(output) => Self::serialize_tensor(&output),
                            Err(e) => Err(e)
                        }
                    },
                    Err(_) => Err(anyhow!("Channel receive error"))
                }
            },
            default(Duration::from_secs(5)) => Err(anyhow!("Inference timeout")),
        }
    }

    fn serialize_tensor(tensor: &Tensor) -> Result<ml::TensorProto> {
        let shape: Vec<i32> = tensor.size().iter().map(|&d| d as i32).collect();
        let data: Vec<f32> = Vec::<f32>::try_from(tensor)?;
        
        Ok(ml::TensorProto {
            shape,
            data,
            dtype: "f32".into(),
        })
    }

    fn deserialize_tensor(proto: &ml::TensorProto) -> Result<Tensor> {
        let shape: Vec<i64> = proto.shape.iter().map(|&d| d as i64).collect();
        Ok(Tensor::f_from_slice(&proto.data)?.reshape(&shape))
    }

    fn deserialize_tensor_request(proto: &ml::TensorRequest) -> Result<Tensor> {
        let shape: Vec<i64> = proto.shape.iter().map(|&d| d as i64).collect();
        Ok(Tensor::f_from_slice(&proto.data)?.reshape(&shape))
    }
}

// Распределённый пул воркеров
struct DistributedWorkerPool {
    workers: Vec<std::thread::JoinHandle<()>>,
    task_sender: crossbeam_channel::Sender<(ml::TensorRequest, crossbeam_channel::Sender<ml::TensorProto>)>,
}

impl DistributedWorkerPool {
    fn new(endpoints: Vec<String>, concurrency: usize) -> Self {
        let (task_sender, task_receiver) = unbounded::<(ml::TensorRequest, crossbeam_channel::Sender<ml::TensorProto>)>();
        let receiver = Arc::new(std::sync::Mutex::new(task_receiver));
        
        let workers = (0..concurrency).map(|_id| {
            let receiver = receiver.clone();
            let endpoints = endpoints.clone();
            
            std::thread::spawn(move || {
                let mut rng = rand::thread_rng();
                let rt = tokio::runtime::Runtime::new().unwrap();
                let client = reqwest::Client::new();
                
                while let Ok((request, result_sender)) = receiver.lock().unwrap().recv() {
                    // Выбор случайного эндпоинта
                    let endpoint = &endpoints[rng.gen_range(0..endpoints.len())];
                    let url = format!("{}/predict", endpoint);
                    
                    // Отправка на удалённый сервер
                    rt.block_on(async {
                        if let Ok(response) = client.post(&url)
                            .json(&request)
                            .send().await
                        {
                            if let Ok(result) = response.json::<ml::TensorProto>().await {
                                let _ = result_sender.send(result);
                            }
                        }
                    });
                }
            })
        }).collect();
        
        Self { workers, task_sender }
    }
    
    fn predict(&self, request: ml::TensorRequest) -> crossbeam_channel::Receiver<ml::TensorProto> {
        let (sender, receiver) = bounded(1);
        self.task_sender.send((request, sender)).unwrap();
        receiver
    }
}

// Main сервер
#[tokio::main]
async fn main() -> Result<()> {
    // Загрузка модели
    let runtime = OnnxRuntime::new("resnet50.onnx")?;
    let svc = ml::inference_service_server::InferenceServiceServer::new(runtime);
    
    // Запуск gRPC сервера
    tokio::spawn(async move {
        Server::builder()
            .add_service(svc)
            .serve("0.0.0.0:50051".parse().unwrap())
            .await
            .unwrap();
    });
    
    // Создание распределённого пула
    let worker_pool = DistributedWorkerPool::new(
        vec!["http://worker1:50051".into(), "http://worker2:50051".into()],
        8
    );
    
    // Имитация клиентских запросов
    for _ in 0..100 {
        let request = ml::TensorRequest {
            shape: vec![1, 3, 224, 224],
            data: vec![0.5; 3*224*224],
            dtype: "f32".into(),
        };
        
        let receiver = worker_pool.predict(request);
        
        select! {
            recv(receiver) -> result => {
                if let Ok(tensor) = result {
                    println!("Received result: {:?}", tensor.shape);
                }
            },
            default(Duration::from_secs(3)) => eprintln!("Request timeout"),
        }
    }
    
    Ok(())
}
