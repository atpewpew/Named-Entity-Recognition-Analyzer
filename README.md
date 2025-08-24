# Named Entity Recognition (NER) Web Application

A user-friendly Streamlit web application for Named Entity Recognition using both PyTorch BiLSTM and spaCy models.

## Features

- **Dual Model Support**: Compare results from custom PyTorch BiLSTM model and spaCy's pre-trained model
- **Multiple Input Methods**: Type text, upload files, or use sample texts
- **Interactive Visualization**: Beautiful entity highlighting and statistics
- **Model Comparison**: Side-by-side comparison of different models
- **Responsive Design**: Clean, modern UI that works on different screen sizes

## Installation

1. **Clone or download the project files**

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download spaCy English model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

1. **Make sure you have the trained model file**:
   - Ensure `ner_model.pth` is in the same directory as `streamlit_app.py`
   - This file should be generated from running the Jupyter notebook

2. **Run the Streamlit app**:
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Open your browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

## How to Use the App

1. **Choose Input Method**:
   - **Type text**: Enter your text directly
   - **Upload file**: Upload a .txt file
   - **Use sample text**: Select from predefined examples

2. **Configure Analysis Options**:
   - Enable/disable PyTorch model results
   - Enable/disable spaCy model results
   - Enable model comparison view

3. **Analyze**: Click the "Analyze Text" button to process your text

4. **View Results**:
   - See detected entities in tabular format
   - View statistics and metrics
   - Enjoy interactive visualizations

## Model Information

- **PyTorch Model**: Custom BiLSTM model trained on your NER dataset
- **spaCy Model**: Pre-trained English model (`en_core_web_sm`)
- **Device Support**: Automatically uses GPU if available

## File Structure

```
NER/
├── streamlit_app.py      # Main Streamlit application
├── requirements.txt      # Python dependencies
├── ner_model.pth        # Trained PyTorch model (generated from notebook)
├── ner_dataset.csv      # Training dataset
├── ner.ipynb           # Jupyter notebook for training
└── README.md           # This file
```

## Troubleshooting

1. **Model not found error**: Ensure `ner_model.pth` exists and is in the correct directory
2. **spaCy model error**: Run `python -m spacy download en_core_web_sm`
3. **CUDA issues**: The app automatically falls back to CPU if GPU is not available
4. **Package conflicts**: Use a virtual environment and install exact versions from requirements.txt

## Customization

You can easily customize the app by:
- Modifying the CSS styles in the `st.markdown()` sections
- Adding new sample texts
- Implementing additional NER models
- Extending the comparison functionality
- Adding export/save features

## Technologies Used

- **Streamlit**: Web application framework
- **PyTorch**: Deep learning framework for custom NER model
- **spaCy**: Natural language processing library
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing

Enjoy exploring Named Entity Recognition with this interactive web application! 🚀
