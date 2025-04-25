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
        if len(predicted) == 1 and predicted[0] == "Other : Other":
            narr_final_joined = "Other"
            subnarr_final_joined = "Other"
        else:
            narr_final = []
            subnarr_final = []
            for i in range(len(predicted)):
                narr, subnarr = predicted[i].split(" : ")
                narr_final.append(cat_code + ": " + narr)
                subnarr_final.append(cat_code + ": " + narr + ": " + subnarr)
            narr_final_joined = ";".join(narr_final)
            subnarr_final_joined = ";".join(subnarr_final)
        
        annotations = {}
        annotations.setdefault("Narrative", []).append(
            {
                "start": 0,
                "end": len(request.content),
                "features": {
                    "category": category,
                    "main_narrative": narr_final_joined,
                    "sub_narrative": subnarr_final_joined
                    #"rumour_label": self.labels[predicted_class_id],
                    #"confidence": probabilities[predicted_class_id],
                    #"status": self.label_mapping[self.labels[predicted_class_id]]
                },
            }
        )
        return AnnotationsResponse(annotations=annotations)

flask_service = H3Prompt("h3prompt")
app = flask_service.app
