# Cookbook: Book-Audio-Summary-Generator
- Welcome to an engaging tutorial designed to walk you through building a book audio summary generator, where we use large language model(LLM) with text-to-speech model(TTS) to process PDF books, extract key ideas, quotes, and actionable items, and convert them into engaging audio summaries.
## Architecture
<img width="1339" alt="image" src="https://github.com/user-attachments/assets/b28104b7-1c5f-48bd-bd02-e5cbed4c4d83">

---
## Prerequisites
- **Git**. You would need git installed on your system if you wish to customize the repo after forking.
- **Python>=3.8**. You would need Python to customize the code in the app.py according to your needs.
- **Curl**. You would need Curl if you want to make API calls from the terminal itself.

---
## Quick Start
Here is a quick start to help you get up and running with this template on Inferless.

### Fork the Repository
Get started by forking the repository. You can do this by clicking on the fork button in the top right corner of the repository page.

This will create a copy of the repository in your own GitHub account, allowing you to make changes and customize it according to your needs.

### Create a Custom Runtime in Inferless
To access the custom runtime window in Inferless, simply navigate to the sidebar and click on the Create new Runtime button. A pop-up will appear.

Next, provide a suitable name for your custom runtime and proceed by uploading the inferless-runtime.yaml file given above. Finally, ensure you save your changes by clicking on the save button.

### Add Your Hugging Face Auth Token
Go into the `inferless.yaml` and  add your hugging face access token. Make sure to check the repo is private to protect your hugging face key.

### Import the Model in Inferless
Log in to your inferless account, select the workspace you want the model to be imported into and click the Add Model button.

Select the PyTorch as framework and choose **Repo(custom code)** as your model source and select your provider, and use the forked repo URL as the **Model URL**.

Enter all the required details to Import your model. Refer [this link](https://docs.inferless.com/integrations/github-custom-code) for more information on model import.

---
## Curl Command
Following is an example of the curl command you can use to make inference. You can find the exact curl command in the Model's API page in Inferless.
```bash
curl --location '<your_inference_url>' \
          --header 'Content-Type: application/json' \
          --header 'Authorization: Bearer <your_api_key>' \
          --data '{
                  "inputs": [
                              {
                                  "name": "book_url",
                                  "shape": [
                                      1
                                  ],
                                  "data": [
                                      "https://github.com/rbgo404/Files/raw/main/LearnGenerativeAI.pdf"
                                  ],
                                  "datatype": "BYTES"
                              }
                          ]  
                }
            '
```
---
## Customizing the Code
Open the `app.py` file. This contains the main code for inference. It has three main functions, initialize, infer and finalize.

**Initialize** -  This function is executed during the cold start and is used to initialize the model. If you have any custom configurations or settings that need to be applied during the initialization, make sure to add them in this function.

**Infer** - This function is where the inference happens. The argument to this function `inputs`, is a dictionary containing all the input parameters. The keys are the same as the name given in inputs. Refer to [input](#input) for more.

**Finalize** - This function is used to perform any cleanup activity for example you can unload the model from the gpu.

For more information refer to the [Inferless docs](https://docs.inferless.com/).
