# SURF-Task

In this task I investigate the intermediate activations for VLMs on counting objects in images.

## Dataset (pixmo_count_dataset.ipynb)

We get our dataset from the [pixmo-count](https://huggingface.co/datasets/allenai/pixmo-count/viewer/default/test) dataset.

Unfortunately, because the test split does not contain counts of 1, I will be taking random samples from the train split.

I have also conducted experiments with all data randomly sampled from the train split and the results are similar, so I feel that using the train split for part of the dataset is acceptable.

I chose 56 images for each count from 1-5, with 280 total images.

## Evaluation

These are the accuracies for all three models, when tested on images with a specific number of objects.

| Model | 1     | 2     | 3     | 4     | 5     | Total |
|-------|-------|-------|-------|-------|-------|-------|
| Qwen  | 98.2% | 92.9% | 80.4% | 71.4% | 76.8% | 83.9% |
| LLaVA | 100%  | 73.2% | 32.1% | 23.2% | 75.0% | 60.7% |
| Molmo | 66.1% | 92.9% | 87.5% | 64.3% | 57.1% | 73.6% |

## Plots

### Qwen:

![alt text](https://github.com/0x0c5cbae8/SURF-Task/raw/main/qwen_plots/qwen_10.png "Layer 10")
![alt text](https://github.com/0x0c5cbae8/SURF-Task/raw/main/qwen_plots/qwen_16.png "Layer 16")
![alt text](https://github.com/0x0c5cbae8/SURF-Task/raw/main/qwen_plots/qwen_17.png "Layer 17")
![alt text](https://github.com/0x0c5cbae8/SURF-Task/raw/main/qwen_plots/qwen_20.png "Layer 20")
![alt text](https://github.com/0x0c5cbae8/SURF-Task/raw/main/qwen_plots/qwen_28.png "Layer 28")

### LLaVA:

![alt text](https://github.com/0x0c5cbae8/SURF-Task/raw/main/llava_plots/llava_5.png "Layer 5")
![alt text](https://github.com/0x0c5cbae8/SURF-Task/raw/main/llava_plots/llava_10.png "Layer 10")
![alt text](https://github.com/0x0c5cbae8/SURF-Task/raw/main/llava_plots/llava_12.png "Layer 12")
![alt text](https://github.com/0x0c5cbae8/SURF-Task/raw/main/llava_plots/llava_19.png "Layer 19")
![alt text](https://github.com/0x0c5cbae8/SURF-Task/raw/main/llava_plots/llava_32.png "Layer 32")

### Molmo:

![alt text](https://github.com/0x0c5cbae8/SURF-Task/raw/main/molmo_plots/molmo_5.png "Layer 5")
![alt text](https://github.com/0x0c5cbae8/SURF-Task/raw/main/molmo_plots/molmo_12.png "Layer 12")
![alt text](https://github.com/0x0c5cbae8/SURF-Task/raw/main/molmo_plots/molmo_16.png "Layer 16")
![alt text](https://github.com/0x0c5cbae8/SURF-Task/raw/main/molmo_plots/molmo_20.png "Layer 20")
![alt text](https://github.com/0x0c5cbae8/SURF-Task/raw/main/molmo_plots/molmo_28.png "Layer 28")

## Discussion

### Before dimensionality reduction, what are the original dimensions of your activations? What does each dimension represent?

- Qwen: (29, 2, \[varies by input, roughly 2500\], 3584)
- LLaVA: (33, 2, \[varies by input, roughly 600\], 4096)
- Molmo: (29, 1, \[varies by input, roughly 1200\], 3584)

In general, the dimensions represent (layer_count+1, batch_size, sequence_length, embedding_length).

The +1 from layer_count is because the "0-th" layer is the encoded input tokens, and the i-th index (i>0) is the output of the i-th layer.

Interestingly, the higher the token count, the higher the average accuracy.

### At what sequence index/indices did you acquire the intermediate activations, and why?

(5 intermediate layers, :, -1, :)

I took the last element because that corresponds to the next predicted token to be outputted.

### Try a few different approaches. Think about the pros and cons of each approach. 

I noticed that Qwen outputs the token containing the number after just one pass.

However, LLaVA always outputs a blank space token (id 29871) and then outputs the number.

And, Molmo outputs the space token (id 220) and then outputs the number.

Thus, for LLaVA and Molmo, I added the corresponding tokens to the end of the input sequence to force the models to
output the number on their next pass. The results showed better separation.

### What is the difference between different layers?

The separation between activations from inputs of different counts gets bigger as the model progresses through the layers.

Additionally, for all three models, it seems that the activations were first separated into two distinct chunks initially.
LLaVA's and Molmo's activation distributions showed this most clearly.

I believe that this separation is due to the different labels in the image. One cluster is counting the number of people,
the other is counting other objects. However, I have not experimented on the hypothesis yet.

### Number of layers in each language model

- Qwen: 28 Layers
- LLaVA: 32 Layers
- Molmo: 28 Layers
