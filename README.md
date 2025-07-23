# 🔧 AI Field Technician Assistant

A comprehensive multi-modal AI agent for equipment diagnosis and maintenance support. This application combines computer vision, natural language processing, and time-series analysis to assist field technicians in diagnosing equipment issues.

## 🌟 Features

### Multi-Modal Analysis
- **Visual Inspection**: Analyze equipment photos using GPT-4V to identify visible issues
- **Sensor Data Analysis**: Real-time monitoring and anomaly detection
- **Knowledge Base Search**: Access to technical documentation and procedures
- **Maintenance History**: Track past issues and solutions

### Core Capabilities
- Equipment diagnosis using multiple data sources
- Predictive maintenance recommendations
- Step-by-step repair procedures
- Safety guidelines and considerations
- Historical trend analysis

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key
- 4GB+ RAM recommended

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/iamkira-ai/multi-modal-field-tech-agent.git
cd multi-modal-field-tech-agent
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```

4. **Run the application**
```bash
streamlit run field_technician_ai.py
```

5. **Access the app**
Open http://localhost:8501 in your browser

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)
```bash
# Set your OpenAI API key in .env file
echo "OPENAI_API_KEY=your_key_here" > .env

# Start the application
docker-compose up -d
```

### Using Docker directly
```bash
docker build -t field-technician-ai .
docker run -p 8501:8501 -e OPENAI_API_KEY=your_key_here field-technician-ai
```

## 📊 Usage Examples

### 1. Equipment Diagnosis
- Select equipment from the sidebar
- Describe the issue (e.g., "unusual noise from pump")
- Optionally upload a photo
- Click "Start Diagnosis" for AI analysis

### 2. Sensor Monitoring
- View real-time sensor data trends
- Identify anomalies and patterns
- Get alerts for unusual readings

### 3. Maintenance Planning
- Review maintenance history
- Get recommendations for preventive maintenance
- Track parts usage and costs

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │────│  LangChain Agent │────│   OpenAI APIs   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                       │
         │                        │                       │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Image Analysis │    │   Knowledge Base │    │  Sensor Data    │
│    (GPT-4V)     │    │     (FAISS)      │    │   (SQLite)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🛠️ Technical Components

### AI Models
- **GPT-4**: Text analysis and reasoning
- **GPT-4V**: Visual inspection and image analysis
- **FAISS**: Vector similarity search for knowledge retrieval

### Data Sources
- Real-time sensor data (temperature, pressure, vibration, power)
- Equipment photos and visual inspections
- Maintenance history and technician notes
- Technical documentation and procedures

### Tools & Integrations
- **LangChain**: Agent orchestration and tool coordination
- **Streamlit**: Interactive web interface
- **SQLite**: Local data storage
- **Plotly**: Data visualization
- **PIL**: Image processing

## 📈 Use Cases

### Manufacturing
- Production line equipment monitoring
- Quality control inspections
- Predictive maintenance scheduling

### Facilities Management
- HVAC system diagnostics
- Electrical equipment monitoring
- Building maintenance optimization

### Industrial Operations
- Heavy machinery troubleshooting
- Safety compliance monitoring
- Operational efficiency analysis

## 🔧 Customization

### Adding New Equipment Types
1. Update `default_equipment_types` in `config.py`
2. Add relevant documentation to knowledge base
3. Configure sensor mappings in database

### Extending Analysis Capabilities
1. Create new LangChain tools in `field_technician_ai.py`
2. Add custom analysis functions
3. Update agent configuration

### Custom Knowledge Base
1. Replace sample documentation in `KnowledgeBase.load_knowledge_base()`
2. Add your technical manuals and procedures
3. Update embedding model if needed

## 🧪 Testing

Run the test suite:
```bash
python -m pytest test_application.py -v
```

## 📋 API Reference

### Core Classes

#### `FieldTechnicianAgent`
Main orchestration class combining all capabilities.

#### `EquipmentDatabase`
Manages equipment data and sensor readings.

#### `KnowledgeBase`
Handles technical documentation search.

#### `ImageAnalyzer`
Processes equipment images using computer vision.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 🎯 Roadmap

- [ ] Mobile app for field technicians
- [ ] Integration with popular CMMS systems
- [ ] Advanced predictive analytics
- [ ] Multi-language support
- [ ] Offline mode capabilities
- [ ] Voice interaction support

---

**Made with ❤️ for the maintenance and reliability community**
