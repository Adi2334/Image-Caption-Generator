import nltk
import numpy as np
import pickle
import os
from utils import data_loader as DL
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sacrebleu.metrics import BLEU
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

nltk.download('wordnet')

def evaluate_metrics(pred_captions, true_captions):
    """
    Evaluate image captioning model using BLEU, METEOR, ROUGE, CIDEr, SPICE, and BERTScore.

    :param pred_captions: Dict {image_id: predicted caption}
    :param true_captions: Dict {image_id: [list of reference captions]}
    """

    # Convert to list format
    references = [true_captions[img] for img in pred_captions.keys()]
    hypotheses = [pred_captions[img] for img in pred_captions.keys()]

    # BLEU Scores
    smoothing = SmoothingFunction().method1
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)

    # METEOR Score
    meteor_scores = []
    for hyp, refs in zip(hypotheses, references):
        meteor_scores.append(nltk.translate.meteor_score.meteor_score(refs, hyp))
    meteor = np.mean(meteor_scores)

    # ROUGE Score
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = [rouge.score(" ".join(hyp), " ".join(refs[0]))['rougeL'].fmeasure for hyp, refs in zip(hypotheses, references)]
    rouge_l = np.mean(rouge_scores)

    # CIDEr Score
    cider_scorer = Cider()
    cider, _ = cider_scorer.compute_score({i: refs for i, refs in enumerate(references)},
                                          {i: [hyp] for i, hyp in enumerate(hypotheses)})

    # SPICE Score
    spice_scorer = Spice()
    spice, _ = spice_scorer.compute_score({i: refs for i, refs in enumerate(references)},
                                          {i: [hyp] for i, hyp in enumerate(hypotheses)})

    # BERTScore
    P, R, F1 = bert_score(hypotheses, [refs[0] for refs in references], lang="en", rescale_with_baseline=True)

    print("Evaluation Results:")
    print(f"BLEU-1: {bleu1:.4f}")
    print(f"BLEU-2: {bleu2:.4f}")
    print(f"BLEU-3: {bleu3:.4f}")
    print(f"BLEU-4: {bleu4:.4f}")
    print(f"METEOR: {meteor:.4f}")
    print(f"ROUGE-L: {rouge_l:.4f}")
    print(f"CIDEr: {cider:.4f}")
    print(f"SPICE: {spice:.4f}")
    print(f"BERTScore F1: {np.mean(F1):.4f}")

    return {
        "BLEU-1": bleu1, "BLEU-2": bleu2, "BLEU-3": bleu3, "BLEU-4": bleu4,
        "METEOR": meteor, "ROUGE-L": rouge_l, "CIDEr": cider, "SPICE": spice,
        "BERTScore-F1": np.mean(F1)
    }

'''
# Example Usage
true_captions = {
    "img1.jpg": ["a cat sitting on a table", "a cat is on a wooden table"],
    "img2.jpg": ["a man riding a bicycle", "a person is biking on the road"]
}

pred_captions = {
    "img1.jpg": "a cat sitting on the table",
    "img2.jpg": "a person riding a bike"
}
'''
if __name__ == '__main__':

    dataset_text = "/home/adi/img_cap_gen/Data/Flickr8k_text"
    features_file = "features_aug.p"

    # Load test data
    test_filename = os.path.join(dataset_text, "Flickr_8k.testImages.txt")
    test_imgs = DL.load_photos(test_filename)
    true_captions = DL.load_clean_descriptions("descriptions.txt", test_imgs)

    with open("test_captions.p", "rb") as file:
        pred_captions = pickle.load(file)

    evaluate_metrics(pred_captions, true_captions)
