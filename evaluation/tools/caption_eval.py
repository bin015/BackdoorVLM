import subprocess
import tempfile
from pathlib import Path

from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.tokenizer import ptbtokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


class WritablePTBTokenizer:
    """Run the COCO tokenizer with its input file in a writable temp directory."""

    def tokenize(self, captions_for_image):
        image_ids = []
        sentences = []
        for image_id, captions in captions_for_image.items():
            for caption in captions:
                image_ids.append(image_id)
                sentences.append(caption["caption"].replace("\n", " "))

        jar = Path(ptbtokenizer.__file__).with_name(ptbtokenizer.STANFORD_CORENLP_3_4_1_JAR)
        with tempfile.TemporaryDirectory() as directory:
            input_file = Path(directory) / "captions.txt"
            input_file.write_text("\n".join(sentences), encoding="utf-8")
            completed = subprocess.run(
                ["java", "-cp", str(jar), "edu.stanford.nlp.process.PTBTokenizer",
                 "-preserveLines", "-lowerCase", str(input_file)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            )

        lines = completed.stdout.splitlines()
        if len(lines) != len(image_ids):
            raise RuntimeError(f"COCO tokenizer produced {len(lines)} lines for {len(image_ids)} captions")
        tokenized = {}
        for image_id, line in zip(image_ids, lines):
            words = (word for word in line.split() if word not in ptbtokenizer.PUNCTUATIONS)
            tokenized.setdefault(image_id, []).append(" ".join(words))
        return tokenized


class SimpleCOCOEvalCap(COCOEvalCap):
    def __init__(self, coco, cocoRes):
        super().__init__(coco, cocoRes)
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
        ]
        
    def evaluate(self):
        imgIds = self.params['image_id']
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.cocoRes.imgToAnns[imgId]

        tokenizer = WritablePTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)
            
        for scorer, method in self.scorers:
            # print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    # print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                # print("%s: %0.3f"%(method, score))
        self.setEvalImgs()
        
        return {k: float(v) for k, v in self.eval.items()}
