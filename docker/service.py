from elg import FlaskService
from elg.model import AnnotationsResponse 

import demo

class H3Prompt(FlaskService): 

    narratives = demo.load_json('Dataset/labels.json')
    main_narratives_with_explanations = demo.load_json('Dataset/main_narratives_with_explanations.json')
    sub_narratives_with_explanations = demo.load_json('Dataset/sub_narratives_with_explanations.json')
    model, tokenizer = demo.load_model("Model")

    cat_mapping = {"Ukraine-Russia War":"URW", "Climate Change":"CC", "Other":"Other"}

    def process_text(self, request):
        category, predicted = demo.classify_document(request.content, self.model, self.tokenizer, self.narratives, self.main_narratives_with_explanations, self.sub_narratives_with_explanations)
        if "climate" in category.lower():
            category = "Climate Change"
        if "ukraine" in category.lower() or "russia" in category.lower():
            category = "Ukraine-Russia War"
        cat_code = self.cat_mapping[category]

        annotations = {}
        annotations.setdefault("Narrative", [])

        if len(predicted) == 1 and predicted[0] == "Other : Other":
            annotations.setdefault("Narrative", []).append({
                "start": 0,
                "end": len(request.content),
                "features": {
                    "category": category,
                    "main_narrative": "Other",
                    "sub_narrative": "Other"
                },
            })
        else:
            narr_final = []
            subnarr_final = []
            for i in range(len(predicted)):
                narr, subnarr = predicted[i].split(" : ")
                annotations.setdefault("Narrative", []).append({
                    "start": 0,
                    "end": len(request.content),
                    "features": {
                        "category": category,
                        "main_narrative": narr,
                        "sub_narrative": subnarr
                    },
                })
        
        return AnnotationsResponse(annotations=annotations)

flask_service = H3Prompt("h3prompt")
app = flask_service.app
