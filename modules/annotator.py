import supervision as sv

class Annotator():

    def __init__(self, scale_factor, do_labels):
        
        self.bb_annotator = sv.BoxAnnotator(
            thickness = int(3 * scale_factor)  # Scale the box thickness
        )

        self.do_labels = do_labels
        if do_labels:
            self.lbl_annotator = sv.LabelAnnotator(
                text_thickness = int(2 * scale_factor),
                text_scale = 1 * scale_factor,
                text_padding =  int(4 * scale_factor)
            )

    def __call__(self, image, detections, labels):
        
        annotated_frame = self.bb_annotator.annotate(scene=image.copy(), detections=detections)
        if self.do_labels: annotated_frame = self.lbl_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        return annotated_frame