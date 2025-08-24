import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import spacy
from spacy import displacy
import re
from io import StringIO

# --- Streamlit compatibility helpers ---
def _safe_rerun():
    """Rerun the app in a way that works across Streamlit versions."""
    # Newer versions
    if hasattr(st, "rerun"):
        return st.rerun()
    # Backward compatibility
    if hasattr(st, "experimental_rerun"):
        return st.experimental_rerun()
    # Fallback: advise manual reload
    st.warning("Please reload the page to clear results.")

# Set page config
st.set_page_config(
    page_title="NER Model Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .entity-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
        text-align: center;
    }
    .stTextArea > div > div > textarea {
        border-radius: 10px;
        border: 2px solid #e1e5eb;
        font-size: 16px;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #0d5aa7;
        box-shadow: 0 5px 10px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# PyTorch model class (same as in notebook)
class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, tagset_size, num_layers=1):
        super(BiLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            bidirectional=True, 
            dropout=0.2 if num_layers > 1 else 0,
            batch_first=True
        )
        self.lstm = nn.LSTM(
            hidden_dim * 2,
            hidden_dim,
            num_layers=1,
            dropout=0.5,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.5)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        
    def forward(self, sentence):
        embeds = self.embedding(sentence)
        bilstm_out, _ = self.bilstm(embeds)
        lstm_out, _ = self.lstm(bilstm_out)
        lstm_out = self.dropout(lstm_out)
        tag_space = self.hidden2tag(lstm_out)
        return tag_space

@st.cache_resource
def load_model():
    """Load the trained PyTorch model and mappings"""
    try:
        # Determine device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model checkpoint
        checkpoint = torch.load('ner_model.pth', map_location=device)
        
        # Create model with saved parameters
        model = BiLSTMModel(
            vocab_size=checkpoint['input_dim'],
            embedding_dim=checkpoint['output_dim'],
            hidden_dim=checkpoint['output_dim'],
            tagset_size=checkpoint['n_tags']
        )
        
        # Load state dict and move to device
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # Ensure all model parameters are on the same device
        for param in model.parameters():
            param.data = param.data.to(device)
        
        st.success(f"✅ Model loaded successfully on {device}")
        
        return model, checkpoint, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please make sure 'ner_model.pth' exists in the current directory and was generated correctly.")
        return None, None, None

@st.cache_resource
def load_spacy_model():
    """Load spaCy model, attempting on-demand install if missing (for cloud)."""
    try:
        return spacy.load('en_core_web_sm')
    except OSError:
        try:
            import spacy.cli as spacy_cli
            with st.spinner("Downloading spaCy model en_core_web_sm (first run only)..."):
                spacy_cli.download("en_core_web_sm")
            return spacy.load('en_core_web_sm')
        except Exception as e:
            st.error(f"Failed to load/download spaCy model: {e}")
            st.info("If running on Streamlit Cloud, ensure the model is included in requirements.txt.")
            return None

def predict_entities(model, text, token2idx, idx2tag, device, max_len=104):
    """Predict entities using the trained model"""
    # Simple tokenization and preprocessing
    words = text.strip().split()
    
    if not words:
        return []
    
    # Convert words to indices
    word_indices = []
    for word in words:
        # Clean the word (remove punctuation, convert to lowercase)
        clean_word = word.lower().strip('.,!?;:"()[]{}')
        if clean_word in token2idx:
            word_indices.append(token2idx[clean_word])
        else:
            # Use a default index for unknown words (UNK token)
            word_indices.append(len(token2idx) - 1)
    
    # Pad or truncate to max_len
    if len(word_indices) < max_len:
        word_indices.extend([len(token2idx) - 1] * (max_len - len(word_indices)))
    else:
        word_indices = word_indices[:max_len]
    
    # Convert to tensor and ensure it's on the correct device
    input_tensor = torch.tensor([word_indices], dtype=torch.long)
    input_tensor = input_tensor.to(device)
    
    # Ensure model is on the correct device
    model = model.to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        try:
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 2)
            
            # Move predictions back to CPU for processing
            predicted = predicted.cpu()
            
        except RuntimeError as e:
            st.error(f"Runtime error during prediction: {str(e)}")
            return []
    
    # Convert predictions back to tags
    predicted_tags = []
    for i, word in enumerate(words[:len(predicted[0])]):  # Only process actual words
        if i < len(predicted[0]):
            tag_idx = predicted[0][i].item()
            if tag_idx in idx2tag:
                predicted_tags.append(idx2tag[tag_idx])
            else:
                predicted_tags.append('O')
        else:
            predicted_tags.append('O')
    
    return list(zip(words[:len(predicted_tags)], predicted_tags))

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Named Entity Recognition Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.markdown("## 🛠️ Model Information")
    st.sidebar.info("This app uses a BiLSTM model trained with PyTorch for Named Entity Recognition.")
    
    # Load models
    pytorch_model, checkpoint, device = load_model()
    spacy_model = load_spacy_model()
    
    if pytorch_model is None:
        st.error("❌ Could not load the PyTorch model. Please ensure 'ner_model.pth' exists in the current directory.")
        st.stop()
    
    # Model info in sidebar
    if checkpoint:
        st.sidebar.markdown("### Model Details")
        st.sidebar.write(f"**Vocabulary Size:** {checkpoint['input_dim']:,}")
        st.sidebar.write(f"**Number of Tags:** {checkpoint['n_tags']}")
        st.sidebar.write(f"**Device:** {device}")
        
        # Display tag mappings
        if st.sidebar.expander("View Tag Mappings"):
            st.sidebar.json(checkpoint['idx2tag'])
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📝 Enter Text for Analysis</h2>', unsafe_allow_html=True)
        
        # Text input options
        input_method = st.radio(
            "Choose input method:",
            ("Type text", "Upload file", "Use sample text"),
            horizontal=True
        )
        
        sample_texts = [
            "Hi, My name is John Smith. I work at Google in New York.",
            "Apple Inc. was founded by Steve Jobs in California.",
            "Barack Obama was the President of the United States.",
            "Microsoft is headquartered in Redmond, Washington."
        ]
        
        if input_method == "Type text":
            user_input = st.text_area(
                "Enter your text here:",
                height=150,
                placeholder="Type or paste your text here for entity recognition..."
            )
        elif input_method == "Upload file":
            uploaded_file = st.file_uploader("Choose a text file", type=['txt'])
            if uploaded_file is not None:
                user_input = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
                st.text_area("File content:", user_input, height=150, disabled=True)
            else:
                user_input = ""
        else:  # Sample text
            selected_sample = st.selectbox("Choose a sample text:", sample_texts)
            user_input = selected_sample
            st.text_area("Selected text:", user_input, height=100, disabled=True)
    
    with col2:
        st.markdown('<h2 class="sub-header">⚙️ Options</h2>', unsafe_allow_html=True)
        
        # Analysis options
        show_pytorch = st.checkbox("Show PyTorch Model Results", value=True)
        show_spacy = st.checkbox("Show spaCy Results", value=True)
        show_comparison = st.checkbox("Show Model Comparison", value=False)
        
        st.markdown("### Analysis Controls")
        analyze_button = st.button("🔍 Analyze Text", type="primary")
        
        if st.button("🗑️ Clear Results"):
            _safe_rerun()
    
    # Analysis section
    if analyze_button and user_input.strip():
        st.markdown("---")
        st.markdown('<h2 class="sub-header">📊 Analysis Results</h2>', unsafe_allow_html=True)
        
        # PyTorch Model Results
        if show_pytorch and checkpoint:
            with st.spinner("Analyzing with PyTorch model..."):
                try:
                    # Debug information
                    st.write(f"🔧 Debug Info: Using device {device}")
                    st.write(f"🔧 Model device: {next(pytorch_model.parameters()).device}")
                    
                    predictions = predict_entities(
                        pytorch_model, 
                        user_input, 
                        checkpoint['token2idx'], 
                        checkpoint['idx2tag'], 
                        device
                    )
                    
                    st.markdown("### 🤖 PyTorch BiLSTM Model Results")
                    
                    # Display results in a nice format
                    pytorch_entities = [(word, tag) for word, tag in predictions if tag != 'O']
                    
                    if pytorch_entities:
                        # Create a DataFrame for better display
                        df_pytorch = pd.DataFrame(pytorch_entities, columns=['Word', 'Entity Type'])
                        st.dataframe(df_pytorch, use_container_width=True)
                        
                        # Show statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Entities", len(pytorch_entities))
                        with col2:
                            st.metric("Unique Types", len(set([tag for _, tag in pytorch_entities])))
                        with col3:
                            entity_types = [tag for _, tag in pytorch_entities]
                            most_common = max(set(entity_types), key=entity_types.count) if entity_types else "None"
                            st.metric("Most Common Type", most_common)
                            
                        # Show all predictions (including O tags) for debugging
                        with st.expander("View All Predictions (Debug)"):
                            debug_df = pd.DataFrame(predictions, columns=['Word', 'Predicted Tag'])
                            st.dataframe(debug_df, use_container_width=True)
                            
                    else:
                        st.info("No named entities found by PyTorch model.")
                        
                        # Show all predictions for debugging when no entities found
                        with st.expander("View All Predictions (Debug)"):
                            debug_df = pd.DataFrame(predictions, columns=['Word', 'Predicted Tag'])
                            st.dataframe(debug_df, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error with PyTorch model: {str(e)}")
                    st.error("Please check the console for more details.")
                    # Print detailed error for debugging
                    import traceback
                    st.code(traceback.format_exc())
        
        # spaCy Model Results
        if show_spacy and spacy_model:
            with st.spinner("Analyzing with spaCy model..."):
                try:
                    doc = spacy_model(user_input)
                    
                    st.markdown("### 🎯 spaCy Model Results")
                    
                    if doc.ents:
                        # Create DataFrame for spaCy results
                        spacy_entities = [(ent.text, ent.label_, spacy.explain(ent.label_)) for ent in doc.ents]
                        df_spacy = pd.DataFrame(spacy_entities, columns=['Word', 'Entity Type', 'Description'])
                        st.dataframe(df_spacy, use_container_width=True)
                        
                        # Show statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Entities", len(doc.ents))
                        with col2:
                            st.metric("Unique Types", len(set([ent.label_ for ent in doc.ents])))
                        with col3:
                            entity_labels = [ent.label_ for ent in doc.ents]
                            most_common = max(set(entity_labels), key=entity_labels.count) if entity_labels else "None"
                            st.metric("Most Common Type", most_common)
                        
                        # Visualization
                        st.markdown("### 🎨 Entity Visualization")
                        try:
                            html = displacy.render(doc, style="ent", jupyter=False)
                            st.components.v1.html(html, height=300)
                        except:
                            st.info("Visualization not available for this text.")
                    else:
                        st.info("No named entities found by spaCy model.")
                        
                except Exception as e:
                    st.error(f"Error with spaCy model: {str(e)}")
        
        # Model Comparison
        if show_comparison and show_pytorch and show_spacy:
            st.markdown("### ⚖️ Model Comparison")
            st.info("This feature compares the results from both models. Implementation can be extended based on specific requirements.")
    
    elif analyze_button:
        st.warning("⚠️ Please enter some text to analyze.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 1rem;'>
            Built with ❤️ using Streamlit, PyTorch, and spaCy
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
