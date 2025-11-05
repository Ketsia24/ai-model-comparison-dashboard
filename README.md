<img width="947" height="391" alt="image" src="https://github.com/user-attachments/assets/119b4078-0ed0-49e3-8967-52618597e536" />
<img width="900" height="389" alt="image" src="https://github.com/user-attachments/assets/72338aeb-3d06-408c-a4a3-672ef5736dc2" />
<img width="890" height="397" alt="image" src="https://github.com/user-attachments/assets/e2d79fad-976b-43cc-adbf-51c6aaf427c3" />
<img width="884" height="362" alt="image" src="https://github.com/user-attachments/assets/bb705fda-1c61-4feb-9bae-2a474ae86e11" />
<img width="848" height="413" alt="image" src="https://github.com/user-attachments/assets/c6f70c26-da8a-40e9-8b4e-9104e51da8ca" />

# AI Model Comparison Dashboard

A comprehensive web application for analyzing and comparing Large Language Model (LLM) performance across multiple dimensions including quality, cost, energy efficiency, and response time.

##  Overview

This dashboard provides an intuitive interface to compare AI language models across 30 diverse tasks, helping users make informed decisions about model selection based on their specific needs and constraints.

### Key Features

- **Multi-dimensional Analysis**: Compare models across quality, cost, energy consumption, and response time
- **Interactive Visualizations**: Dynamic charts and graphs with interpretation guides
- **Educational Interface**: Built-in explanations for technical terms and metrics
- **Professional Design**: Clean, emoji-free interface suitable for business presentations
- **Real-time Filtering**: Filter by models and task categories
- **Data Export**: Download analysis results in CSV format

##  Dashboard Sections

### 1. Overview
- Key performance indicators (KPIs)
- Performance summaries by model and task category
- Multi-dimensional radar chart comparison

### 2. Charts
- Quality vs Energy consumption analysis
- Quality vs Cost trade-offs
- Response time by model size
- Performance metric distributions

### 3. Comparison Table
- Overall model rankings
- Task-by-task performance comparison
- Color-coded performance indicators
- Category-wise performance analysis

### 4. Recommendations
- Use case recommendations (Simple, Moderate, Complex tasks)
- Strategic deployment advice
- Green AI best practices
- Key insights and trends

##  Getting Started

### Prerequisites

- Python 3.9+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-model-comparison-dashboard.git
cd ai-model-comparison-dashboard
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

5. Open your browser and navigate to `http://localhost:8502`

## 📁 Project Structure

```
ai-model-comparison-dashboard/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── runs_tasks_01_30.xlsx      # Dataset with model performance data
├── .streamlit/
│   └── config.toml            # Streamlit configuration
└── README.md                  # Project documentation
```

##  Data

The dashboard analyzes performance data from various AI language models including:

- **Models**: Google Gemma, Meta LLaMA, Mistral AI, OpenAI GPT, DeepSeek
- **Model Sizes**: 8B, 20B, 70B+ parameters
- **Tasks**: 30 diverse tasks across different categories
- **Metrics**: Quality scores, response time, energy consumption, CO₂ emissions

### Data Format

The application expects an Excel file with the following columns:
- `task_id`: Unique task identifier
- `task_title`: Task description
- `task_category`: Task category
- `model_name`: AI model name
- `quality_score_1to5`: Quality rating (1-5 scale)
- `latency_sec`: Response time in seconds
- `energy_kwh`: Energy consumption in kWh
- `co2_kg`: CO₂ emissions in kg

##  Technologies Used

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive data visualizations
- **NumPy**: Numerical computations
- **OpenPyXL**: Excel file processing

## 🌱 Green AI Focus

This dashboard emphasizes sustainable AI practices by:

- Tracking energy consumption and CO₂ emissions
- Providing eco-friendly model recommendations
- Highlighting efficiency trade-offs
- Promoting awareness of environmental impact

##  Metrics Explained

### Quality Score
Performance rating from 1-5 based on task completion accuracy and relevance.

### Response Time (Latency)
Time taken for the AI model to generate a response, measured in seconds.

### Energy Consumption
Electrical energy required to run the model, measured in kilowatt-hours (kWh).

### Cost
Estimated operational cost in euros, calculated based on energy consumption.

### Efficiency Score
Composite metric combining quality, cost, energy, and speed performance.

##  Deployment

### Streamlit Cloud (Recommended)

1. Push your code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy with one click

### Local Development

```bash
streamlit run app.py
```

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


##  Authors

- Grâce Esther DONG 

##  Acknowledgments

- Green AI Course 2025
- Streamlit community for the excellent framework
- Plotly for interactive visualizations
- All contributors to the open-source libraries used

##  Support

If you encounter any issues or have questions, please:
1. Check the existing issues on GitHub
2. Create a new issue with a detailed description
3. Include your environment details and error messages

---

**Built with ❤️ for sustainable AI development**
