# News Topic Classifier Using BERT

## Objective
Fine‑tune a transformer model (specifically `bert-base-uncased`) to classify news headlines into topic categories using the AG News dataset from Hugging Face.

## Dataset
The project uses the **AG News Dataset** (available via Hugging Face Datasets) which contains news headlines and corresponding topic labels.

## Methodology / Approach
1. **Data Preparation** – load the dataset, tokenize and preprocess headlines with `bert-base-uncased` tokenizer.
2. **Model Fine‑Tuning** – leverage Hugging Face Transformers to fine‑tune the pre‑trained BERT model on the classification task.
3. **Evaluation** – measure performance using accuracy and F1‑score to validate the model's effectiveness.
4. **Deployment** – create a lightweight web interface using Streamlit or Gradio for live interaction and inference.

## Key Results / Observations
- Achieved competitive accuracy and F1-scores on the validation set after fine-tuning.
- Transformer-based models handle news headline classification effectively due to contextual embeddings.
- Deployment with Streamlit/Gradio allows users to input custom headlines and receive topic predictions in real time.

## Skills Gained
- Natural Language Processing (NLP) using Transformer architectures.
- Transfer learning and fine-tuning of pre-trained models.
- Evaluation metrics specific to text classification (accuracy, F1-score).
- Deploying lightweight models for user interaction with Streamlit or Gradio.

## Usage
1. Install dependencies (Hugging Face Transformers, Datasets, Streamlit/Gradio).
2. Run the notebook `News_Topic_Classifier_Using_BERT.ipynb` to train and evaluate the model.
3. Launch the deployment script (e.g., `streamlit run app.py` or similar) for live interaction.

## Repository
Clear repository name and organization: this repo tracks the news topic classifier project under the `taskeen-mustafa786/news_topic_classifier_using_BERT` organization.

---

*This README provides an overview of the project setup and goals for anyone exploring the repository.*