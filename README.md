# Tensorflow Lite and ONNX Conversion for YOLOv10n Models for NSFW Detection in Mobile

A sample Jupyter notebook implementation for training YOLOv10n models and converting them to TensorFlow Lite format for mobile deployment. This project provides steps from dataset preparation to TFLite model conversion, with automatic dataset detection and fallback to one of the default datasets, COCO128. The program was done in Google Colab and as such its designed for execution in cloud environments, only changing datapaths for easier integration to other environments.

## Features

- **Automated Dataset Detection**: Scans for existing YOLO datasets or downloads COCO128 automatically
- **YOLOv10n Training**: Latest YOLO architecture with optimized performance
- **Multi-format Export**: Export to ONNX, TensorFlow SavedModel, and TensorFlow Lite
- **Mobile Optimization**: FP16 quantization and optimization for mobile deployment
- **Model Validation**: Built-in TFLite model testing and validation
- **Google Colab Ready**: Designed for seamless execution in cloud environments

## System Requirements

- Python 3.8 or higher
- PyTorch and TorchVision
- TensorFlow 2.x
- ONNX and related conversion tools
- CUDA support recommended for training

## Installation

The notebook automatically handles dependency installation:

```python
packages = [
    "ultralytics",
    "torch",
    "torchvision", 
    "tensorflow",
    "onnx",
    "onnx2tf",
    "onnxsim",
]
```

## Usage

### Basic Usage

Open and run the Jupyter notebook:

```bash
jupyter notebook yolov10n.ipynb
```

Or execute it directly:

```bash
python yolov10n.ipynb
```

For Google Colab, simply upload the notebook and run all cells.

### Quick Start

The notebook includes a simplified conversion function for immediate testing:

```python
simple_tflite_conversion()
```

This directly converts a pre-trained YOLOv10n model to TensorFlow Lite format without training.

## Dataset Support

### Automatic Dataset Detection

The pipeline automatically scans for existing datasets in common locations:
- `/dataset`
- `/custom_dataset`  
- `/yolo_dataset`
- `/coco128`

or if its done on Google Colab
- `/content/dataset`
- `/content/custom_dataset`  
- `/content/yolo_dataset`
- `/content/coco128`

### Required Dataset Structure

```
dataset/
├── images/
├── labels/
└── dataset.yaml
```

### Automatic Fallback

If no custom dataset is found, the pipeline automatically:
1. Downloads COCO128 dataset
2. Creates appropriate YAML configuration
3. Sets up training environment

## Key Components

### Dataset Preparation
```python
dataset_path, yaml_path = prepare_dataset()
```
- Scans for existing datasets
- Downloads COCO128 if needed
- Creates YAML configuration files

### Model Training
```python
model, results = train_yolov10n(yaml_path, epochs=100, imgsz=640)
```
- Trains YOLOv10n with configurable parameters
- Automatic GPU/CPU detection
- Progress tracking and validation

### Model Export
```python
onnx_path, tf_path = export_model_formats(model)
```
- Exports to ONNX format
- Attempts direct TensorFlow export
- Handles conversion errors

### TensorFlow Lite Conversion
```python
tflite_path = convert_to_tflite(onnx_path)
```
- Converts ONNX to TensorFlow SavedModel
- Applies mobile optimization settings
- Generates quantized TFLite model

### Model Validation
```python
validate_tflite_model(tflite_path)
```
- Tests TFLite model functionality
- Validates input/output tensors
- Performs inference testing

## Configuration Options

### Training Parameters
- **Epochs**: Default 100 (reducible for faster training)
- **Image Size**: 640x640 pixels
- **Batch Size**: 16 (adjustable based on memory)
- **Device**: Automatic GPU/CPU selection

### Export Settings
- **ONNX**: Dynamic shapes disabled for compatibility
- **TensorFlow**: SavedModel format with optimization
- **TFLite**: FP16 quantization for reduced model size

## Output Files

After successful execution:

- **Trained Model**: `/content/yolov10_training/training_experiment/weights/best.pt`
- **ONNX Model**: `best.onnx`
- **TensorFlow SavedModel**: `/content/tflite_models/saved_model/`
- **TensorFlow Lite**: `/content/tflite_models/yolov10n_coco128.tflite`

## Mobile Deployment

The generated TensorFlow Lite model is optimized for mobile deployment:

- **FP16 Quantization**: Reduced model size without significant accuracy loss
- **Mobile Optimizations**: DEFAULT optimization settings applied
- **Typical Size**: 5-15 MB depending on dataset and configuration

### Android Integration
```java
// Load the TFLite model
Interpreter tflite = new Interpreter(loadModelFile());
```

### iOS Integration
```swift
// Load the TFLite model
guard let interpreter = try? Interpreter(modelPath: modelPath) else { return }
```

## Error Handling and Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or image dimensions
2. **Dataset Not Found**: Ensure proper YOLO format structure
3. **ONNX Conversion Fails**: Check model compatibility and dependencies
4. **TFLite Conversion Errors**: Try the simplified conversion method

### Debug Functions

The notebook includes comprehensive error handling:
- Automatic fallback methods for conversion failures
- Detailed error logging and traceback
- Alternative conversion pathways

### Performance Notes

- **Training Time**: 1-2 hours for 50-100 epochs (GPU recommended)
- **Conversion Time**: 5-15 minutes depending on model complexity
- **Memory Requirements**: 4-8 GB RAM, 2-4 GB GPU memory

## Advanced Features

### Custom Dataset Integration
1. Place your YOLO-format dataset in `/content/custom_dataset/`
2. Ensure proper `images/` and `labels/` folder structure
3. Create or modify the YAML configuration file
4. Run the notebook - it will automatically detect and use your dataset

### Batch Processing
The pipeline can handle multiple model exports:
```python
# Export multiple formats simultaneously
formats = ['onnx', 'saved_model', 'tflite']
for format in formats:
    model.export(format=format)
```

## Dependencies

Core packages automatically installed:
- **ultralytics**: YOLOv10 implementation
- **torch/torchvision**: PyTorch framework
- **tensorflow**: TensorFlow and TFLite conversion
- **onnx/onnx2tf**: ONNX format and conversion tools
- **onnxsim**: ONNX model simplification

## Environment Compatibility

- **Google Colab**: Primary target environment
- **Jupyter Notebook**: Local development
- **Kaggle Notebooks**: Cloud training platform
- **Local Python**: Desktop development

## Performance Benchmarks

Typical results on COCO128 dataset:
- **Training**: 50 epochs in ~30 minutes (GPU)
- **mAP@0.5**: 0.6-0.8 depending on dataset
- **Model Size**: 6-12 MB (TFLite with FP16)
- **Inference Speed**: 10-50ms on mobile devices

## Contributing

Contributions are welcome for:
- Additional export formats
- Mobile deployment examples  
- Performance optimizations
- Error handling improvements

## License

This project builds upon the Ultralytics YOLO implementation. Please refer to their licensing terms for commercial usage.

## Support

For issues and questions:
1. Check the error handling sections in the notebook
2. Verify dataset format and structure
3. Ensure all dependencies are properly installed
4. Review TensorFlow Lite documentation for deployment issues

