# Thought process
I started the whole process by doing the research of the available models on Hugging Face. I have found two models - Starcoder and CodeLlama. CodeLlama model was significantly larger and when I attempted to run it, it required substantial resources and computational power. Since I could not run it on Colab (due to GPU limitations), nor locally on my computer, my options narrowed down to tiny-starcoder-py model. Since this model was much smaller, I was able to run it locally, even without a GPU.

Next, I started reviewing the documentation on Hugging Face for tiny-starcoder-py model. At the outset, it was important to understand the meaning of the special tokens <fim_prefix>, <fim_suffix> and <fim_middle>. This was a first step towards the building of model prompts (inputs) and understanding its outputs. After this initial research, I found out that model's inference was really fast and simple which allowed me to quickly obtain results. However, I noticed that the example code provided in the documentation did not yield good predictions and results.

To improve the results, I discovered that it was necessary to configure/fine-tune certain parameters:

```
params = {
    'max_new_tokens': 256,
    'temperature': 0.2,
    'top_k': 50,
    'top_p': 0.1,
    'repetition_penalty': 1.17
}
```

With these adjustments, the example from Hugging Face worked successfully. Given the model's 164 million parameters and the simplicity of the provided examples, it became evident that Tiny Starcoder might struggle with more complex tasks.

Afterwards, I executed some basic Python code example to observe model's capabilities. By doing that, I quickly realized that constructing effective prompts was crucial. Additionally, I noticed that the quality of results varied significantly based on the complexity of the task. This led me to the conclusion that I need to collect Python code from various domains in order to do the evaluation correctly.

# Evaluation process
To gather diverse examples, I collected Python files from my repositories, organizing them into two groups. The first group comprised files from an introductory programming course, which included functions for calculating palindromes, finding the minimum of an array, and others. The second group included files from machine learning courses, covering topics such as classification, GAN networks, clustering and others. 

Next, I have processed these files with a Python [script](https://github.com/vujadinovicn/code-completion-evaluation/blob/main/code_splitter.py). This script contains two functions made for:
- Function extraction (get_functions_code(script_code)):  I used regular expressions to locate function definitions, as well as their start and end, in a single Python file. By doing this, I extracted and collected all functions from my repository's Python files.
- Splitting code (split_function_code(function_code)): I simulated the cursor position and divided the function code into three parts: the prefix, which represents the code before the cursor; the middle, which signifies the missing code that the model is expected to predict; and the suffix, which includes the code following the cursor. I ensured that the prefix and suffix parts of the code took up a minimum of 80% of the total lines, which would usually result in the middle part containing between 1 and 4 lines of code. I also only considered functions that had at least 4 lines in length, as a smaller number of lines does not provide enough information for prediction.

Finally, I iterated through all of my Python files and applied the previous functions to them. I collected all results and saved them to a [JSON file](https://github.com/vujadinovicn/code-completion-evaluation/blob/main/data_for_evaluation/splitted_code.json). This file represented my dataset. 

Now that I had my dataset ready, I was able to perform the evaluation ([code](https://github.com/vujadinovicn/code-completion-evaluation/blob/main/eval.py)). For each example in my dataset, I followed these steps:
- Built a prompt by adding special tokens <fim_prefix>, <fim_suffix> and <fim_middle>
- Passed the prompt through the loaded tiny-starcoder-py model
- Obtained the model prediction
- Collected the following data:
    - The prompt as input.
    - The middle part of the example as ground truth.
    - The generated code as output.
    - The parsed code after the <fim_middle> token in the output as the predicted value.
- Added the previously collected data to a list of processed data.

Lastly, I saved all the processed data (we will call it evaluated data from now on) into [JSON file](https://github.com/vujadinovicn/code-completion-evaluation/blob/main/evaluated_data/all/codes.json).

Finally, I implemented functions for calculating several evaluation metrics, including CHRF, Exact Match, ROUGE, and the Jaccard Index ([code](https://github.com/vujadinovicn/code-completion-evaluation/tree/main/metrics)). I then iterated through the evaluated data to collect the scores for each example. As a result, I created a comprehensive [JSON](https://github.com/vujadinovicn/code-completion-evaluation/blob/main/evaluated_data/all/codes_with_metrics.json) and [csv](https://github.com/vujadinovicn/code-completion-evaluation/blob/main/evaluated_data/all/codes_with_metrics.csv) files containing the evaluated data alongside the corresponding metric scores for each example.

To complete the evaluation, I expanded the previous .csv file to include my own (human) assessment. Each example now contains a column in the .csv file indicating whether the output is correct. Additionally, I reviewed various metric scores and highlighted the most accurate and closest to my evaluation among them. This file is available [here](https://github.com/vujadinovicn/code-completion-evaluation/blob/main/evaluation_report.xlsx).

# Results and discussion
### Positive results
#### File operations
In the image below, we see that the model correctly predicted the missing code lines for functions that load both pickle, JSON and YAML files. These predictions suggest that model is proficient in recognizing standard file-loading patterns for common data. We can conclude the model is well-trained on general file I/O functions in Python and can retrieve file-handling commands when the file type is clearly indicated in the function name. We can say that model is reliable for many file use cases.

![image](https://github.com/user-attachments/assets/7e75b0d9-545b-4e5b-acb4-379479fdd38f)

#### Repetitive structure and patterns
In the image below, we see that the model performed really well in predicting simple, repetitive structures and code - in this code validation_dataset or validation_data_loader are being repeated. The model has great performance in cases where the task was to complete lines following an established code pattern (e.g., initializing a dataset or printing dataset sizes). Additionally, we can conclude that model has  a solid understanding of popular classes, such as Subset and DataLoader, as well as their parameters. Because of this, the model can be used for code completion within the PyTorch framework. These examples also highlights the model's understanding of incremental tasks, where it can replicate patterns when similar structures are presented. This is also the case with the example in the next image:

![image](https://github.com/user-attachments/assets/2d5034cb-ea73-43b6-947f-2d879541c70a)

In the image below, the function has a series of conditional statements. Line generated by the model aligns with the preceding logic. This again suggests the model’s capacity to recognize and extend patterns within conditionally structured code. Model also understood that all numerical values end with digit 4 which helped it to generate number '94'.

![image](https://github.com/user-attachments/assets/533583eb-a077-44b7-941f-78be324c0193)

#### Basic algorithms
In the image below, we see that the model successfully predicts the update of variables, iteration over the elements and variables initialization.

![image](https://github.com/user-attachments/assets/bbea9cbf-1f15-4a03-966e-64aa46d155d0)

### Negative results
#### Complex mathematical and logic tasks
In the image below, we see that the model can recognize the need to derive one of the roots. However, it fails to apply the correct sign for variable b. This indicates potential difficulties with multi-step calculations. Additionally, the model redundantly computes the discriminant, which has already been calculated in the previous line, which suggests model's lack of awareness of the preceding context.

![image](https://github.com/user-attachments/assets/ddcbc5a1-186b-4437-9f19-5d861c12e4fa)

Another example is shown in the image below. While the model understands the function's signature, it makes several significant mistakes in its output. Firstly, it adds an unnecessary elif statement, even though the function would already return a value if the previous condition is true. Additionally, it mistakenly uses != instead of ==, which changes the intended logic. Also, it doesn't output 'return True/False' after the given if statement. Finally, it repeats the condition for checking divisibility by 400, which is a mistake seen in previous examples as well.

![image](https://github.com/user-attachments/assets/b3855d3b-6869-414a-b715-acd88a5e7cf3)

#### ML tasks
In the image below, we see that the model struggles with the prediction and understanding of machine learning code. Although the output includes some of the ground truth code lines, the model failed to initialize and assign the value to prediction and loss variables. These shortcomings suggest that the model may not yet be well-equipped to handle the complexities in machine learning tasks.

However, it is worth noting that the model correctly generates the optimizer.zero_grad() line exclusively for the training function. This is a positive indication, as the optimizer should not be used in the validation function. This highlights the model's awareness of the different contexts in which these operations are applicable. Overall, while the model definitely needs improvement in ML, there are signs of understanding fundamental concepts in this area.

![image](https://github.com/user-attachments/assets/0f78a3b0-7891-47eb-97f6-949ed2391a44)

#### Repetetive code in complex (ML) tasks
In the image below, we see that although the model successfully understands the need to initialize the x1 variable, the value assigned to it is incorrect. It seems like the model has interpreted that input of self.upfeature(x) is requiring a corresponding reverse operation, leading to the assignment of x1 to self.downfeature(x0). In this case, model failed to understand the incremental task, probably because it includes ML code, rather than the basic Python code that we have seen in one of the first examples.

![image](https://github.com/user-attachments/assets/cfb9fb94-540b-4c0d-99a1-245cd1ed8af1)

In the image below, we see that the model recognizes the absence of the self.contract3 layer and correctly assigns its value. However, it goes beyond this by introducing additional layers, even after reaching the point where self.final is defined, which implies that no further layers are needed.

However, the model effectively predicts the forward method, even though that part was not meant to be generated. The inclusion of self.conv1 suggests a tendency to create unnecessary layers, which can be viewed as a form of hallucination in its output. The model struggles to determine where to stop generating code, likely due to the lack of clear definitions regarding the boundaries of the network architecture. In conclusion, even though the model understands some structural aspects, it still needs more training on ML data to generate good results.

![image](https://github.com/user-attachments/assets/4cbc6548-4564-4a3f-9167-2b259bcb32ae)

Lastly, we can see in the image below that model can generate too short of an answer. Like we have already concluded, for more deterministic and accurate results, the model needs more training on this type of data.

![image](https://github.com/user-attachments/assets/24361781-f3be-4b04-9e45-f783772cfa17)

# Metrics
Among the metrics I have used for this evaluation, the best one was the CHRF score, as it aligned most closely with my conclusions. However, there were instances where the Jaccard index provided better results, particularly when the model generated either too little or too much text compared to what was expected. In such cases, CHRF occasionally generated results that were higher than anticipated. In conclusion, I believe that a combination of CHRF and Jaccard would provide the most comprehensive evaluation of the model's performance.

# Conclusion
In conclusion, Tiny Starcoder Py is an excellent model for Python code completion. Its greatest strengths are its speed and compactness which make it a great choice for simpler, repetitive tasks that require quick solutions. However, for some more complex tasks, it definitely requires refinement and more traning.
