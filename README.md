# DLCoG: Dual-Level Code Comment Generation

## Abstract

Code comments that describe the purpose of the code mainly include method comments and snippet comments. Method comments provide a brief description of the method's functionality, helping development team members quickly understand the method's purpose and interface. Snippet comments, on the other hand, detail the internal implementation of the method, helping developers understand the specific role of each code snippet, thus facilitating subsequent maintenance and modifications. 

We manually constructed a high-quality 6.9k Java dataset of <Method, Method Comment, \<Snippet, Snippet Comment>*> based on some open-source Java projects(**raw_dataset/manual_dataset.jsonl**). This dataset was used to train a classification model for identifying "code summaries" in comments, as well as an association model for finding the code corresponding to the comments. A total of 80k multi-level code annotation datasets were built in a larger range of open source projects(**raw_dataset/expansion_dataset.jsonl**).

The paper is under review, please do not disseminate it widely. For reproducibility, we will release more detailed training data and evaluation code upon acceptance.

## Manual Dataset

The original dataset can be found at https://huggingface.co/datasets/bigcode/the-stack/tree/main/data/java

The dataset and eval results are too large to upload to GitHub. You can find the complete code and dataset in https://drive.google.com/drive/folders/1Q7v9PkslmcAXynbhlHm_SL1y9Jkf1NaI?usp=sharing

The format of the dual-level code comment data is as follows:

```json
{
	id:0
	repo:"repo_owner/repo_name"
  path:"root/docA/.../file.java"
  raw_code:"public int fun_name(int arg){...}"
  code_summary:"This function make ..."
  code_snippets:[
  	{
  		sub_id:0
  		code_snippet:"int[] sublist = ..."
  		code_summary:"make a sublist ..."
  		place:(7,9) # Located on lines 7 to 9 of the source code
		},
		...
  ]
}
```

Example:

```json
{
	"repo": "Wangqqingwen/openAGV",
	"path": "opentcs/openTCS-PlantOverview/src/main/java/org/opentcs/guing/components/dialogs/VehiclesPanel.java",
	"star_count": 21,
	"raw_code": "public void setVehicleModels(Collection<VehicleModel> vehicleModels) {\n    // Remove vehicles of the previous model from panel\n    for (SingleVehicleView vehicleView : vehicleViews) {\n      panelVehicles.remove(vehicleView);\n    }\n\n    // Remove vehicles of the previous model from list\n    vehicleViews.clear();\n    // Add vehicles of actual model to list\n    for (VehicleModel vehicle : vehicleModels) {\n      vehicleViews.add(vehicleViewFactory.createSingleVehicleView(vehicle));\n    }\n\n    // Add vehicles of actual model to panel, sorted by name\n    for (SingleVehicleView vehicleView : vehicleViews) {\n      panelVehicles.add(vehicleView);\n    }\n\n    panelVehicles.revalidate();\n  }",
	"code_summary": "/**\n   * Initializes this panel with the current vehicles.\n   *\n   * @param vehicleModels The vehicle models.\n   */",
	"code_snippets": [{
		"sub_id": 0,
		"code_snippet": "for (SingleVehicleView vehicleView : vehicleViews) {\n      panelVehicles.remove(vehicleView);\n    }",
		"code_summary": "// Remove vehicles of the previous model from panel",
		"place": [3, 5]
	}, {
		"sub_id": 1,
		"code_snippet": "vehicleViews.clear();",
		"code_summary": "// Remove vehicles of the previous model from list",
		"place": [8, 8]
	}, {
		"sub_id": 2,
		"code_snippet": "for (VehicleModel vehicle : vehicleModels) {\n  vehicleViews.add(vehicleViewFactory.createSingleVehicleView(vehicle));\n}",
		"code_summary": "// Add vehicles of actual model to list",
		"place": [10, 12]
	}, {
		"sub_id": 3,
		"code_snippet": "for (SingleVehicleView vehicleView : vehicleViews) {\n      panelVehicles.add(vehicleView);\n    }",
		"code_summary": "// Add vehicles of actual model to panel, sorted by name",
		"place": [15, 17]
	}],
	"id": 5044
}
```

We recommend that you conduct research on this manual dataset because it is extracted by human annotations and is of higher quality than the expanded dataset.

The code to expand the dataset can be found in the **expand_dataset** folder.



## Dual-Level Code Comment Generation

You can perform code similarity search, build prompts, and use LLM to get two-level code comments according to the following process.
Encoding your code:

```
python S1_sentence_bert_encode.py
```

Get the similarit data for your code:

```
python s2_get_similarity_data.py
```

Build prompt. The prompt example is shown in the following figure:

```
python s3_build_prompt.py
```

Eval the comment by LLM:

```
python s4_eval.py
```


