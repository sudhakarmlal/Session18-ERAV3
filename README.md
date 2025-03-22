phi2_openassistant_lora
Phi2 model trained on openassistant dataset using 4 bit lora

phi-2 Model
Microsoft Phi-2 is a small language model with 2.7 billion parameters, designed to achieve strong performance despite its compact size. It is built using advanced training techniques and high-quality datasets, making it highly efficient and capable of reasoning, language understanding, and code generation. Phi-2 is part of Microsoft's effort to develop lightweight AI models that offer competitive results while being cost-effective and accessible. Phi-2 has following capabilities: * Phi-2 shows strong reasoning, language understanding, and code generation abilities, competing with much larger models. * It excels in tasks like common sense reasoning, reading comprehension, and math-based problem-solving.

Phi-2 architecture
* Transformer-based: Uses a stack of transformer blocks for deep learning-based text generation and reasoning.
* Decoder-only: Operates as an autoregressive model, predicting tokens sequentially.
* Scaled-down Efficient Design: Optimized to perform well while being significantly smaller than large models like GPT-4.
Parameters:
Training Techniques
Curriculum Learning: Microsoft trained Phi-2 using a structured approach, feeding it high-quality, filtered datasets in stages to improve performance.
Synthetic Data: It incorporates a mix of synthetic and real-world data, including carefully curated textbooks, logical reasoning content, and programming-related datasets.
Heavily Filtered Dataset: Unlike many large models trained on vast, noisy web data, Phi-2 relies on a more refined dataset for better accuracy and safety.
Open Assistant Dataset
The OpenAssistant Dataset (OASST1) is a large-scale, open-source dataset designed to train and fine-tune conversational AI models. Developed by the OpenAssistant project (LAION-AI), it contains high-quality, human-annotated conversations aimed at creating ethical, transparent, and powerful chatbots.

Training Techniques Used
Used Open Assistant Dataset for fine tuning phi-2

Used 4 bit quantization - 4-bit quantization compresses these weights into 4-bit integer representations

LORA Adapter - LoRA is a parameter-efficient fine-tuning technique that adapts only a small subset of model weights rather than modifying the entire model. Rank size used: 8. The layers trained are: q_proj, v_proj. Also dropout used: 0.1 and LORA Alpha: 16

Various hyperparameters used are as below: Batch Size: 4 Learning Rate: 2e-4 Maximum Steps: 5000 Optimizer: paged_adamw_8bit

As I faced OutOfMemory Error during training, after each batch I used to cleanup the GPU and used gc collection torch.cuda.empty_cache() gc.collect()

The LoRA adapter artifacts are stored in google drive

Training Logs:
Step 500: loss: 1.7706 grad_norm: 0.3980 learning_rate: 0.0002 epoch: 0.2668

Step 1000: loss: 1.7892 grad_norm: 0.3363 learning_rate: 0.0002 epoch: 0.5335

Step 1500: loss: 1.7665 grad_norm: 0.3826 learning_rate: 0.0001 epoch: 0.8003

Step 2000: loss: 1.7686 grad_norm: 0.4142 learning_rate: 0.0001 epoch: 1.0667

Step 2500: loss: 1.7430 grad_norm: 0.3755 learning_rate: 0.0001 epoch: 1.3335

Step 3000: loss: 1.7579 grad_norm: 0.4045 learning_rate: 0.0001 epoch: 1.6002

Step 3500: loss: 1.7327 grad_norm: 0.4195 learning_rate: 0.0001 epoch: 1.8670

Step 4000: loss: 1.7414 grad_norm: 0.3816 learning_rate: 0.0000 epoch: 2.1334

Step 4500: loss: 1.7383 grad_norm: 0.4859 learning_rate: 0.0000 epoch: 2.4002

Step 5000: loss: 1.7155 grad_norm: 0.4838 learning_rate: 0.0000 epoch: 2.6669

Step 5000: train_runtime: 4466.2598 train_samples_per_second: 17.9120 train_steps_per_second: 1.1200 train_loss: 1.7524 epoch: 2.6669

HuggingFace Spaces application:
Build the spaces application. Model is loaded by first loading phi-2 model and then merging with LoRA adapter generated after fine tuning. The spaces application can be found in below link:

https://huggingface.co/spaces/sudhakar272/phi2-lora-sft

