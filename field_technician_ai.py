# AI Field Technician Assistant
# A multi-modal agent for equipment diagnosis and maintenance support

import streamlit as st
import openai
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, AgentType
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import numpy as np
import base64
import io
import json
import datetime
import sqlite3
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
import os
from pathlib import Path

# Configuration
st.set_page_config(
    page_title="AI Field Technician Assistant",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

class EquipmentDatabase:
    """Manages equipment data and maintenance history"""
    
    def __init__(self, db_path: str = "equipment.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with sample data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS equipment (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                model TEXT,
                location TEXT,
                install_date DATE,
                last_maintenance DATE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sensor_data (
                id INTEGER PRIMARY KEY,
                equipment_id INTEGER,
                timestamp DATETIME,
                temperature REAL,
                pressure REAL,
                vibration REAL,
                power_consumption REAL,
                FOREIGN KEY (equipment_id) REFERENCES equipment (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS maintenance_logs (
                id INTEGER PRIMARY KEY,
                equipment_id INTEGER,
                date DATE,
                technician TEXT,
                issue_description TEXT,
                resolution TEXT,
                parts_used TEXT,
                FOREIGN KEY (equipment_id) REFERENCES equipment (id)
            )
        ''')
        
        # Insert sample data
        sample_equipment = [
            (1, "Pump A-101", "Centrifugal Pump", "Grundfos CR32-4", "Building A, Level 1", "2020-01-15", "2024-12-01"),
            (2, "Motor B-205", "Electric Motor", "Siemens 1LA7", "Building B, Level 2", "2019-06-20", "2024-11-15"),
            (3, "Compressor C-301", "Air Compressor", "Atlas Copco GA22", "Building C, Level 3", "2021-03-10", "2024-10-30"),
        ]
        
        cursor.executemany('INSERT OR REPLACE INTO equipment VALUES (?,?,?,?,?,?,?)', sample_equipment)
        
        # Generate sample sensor data
        for eq_id in [1, 2, 3]:
            for i in range(100):  # 100 data points per equipment
                timestamp = datetime.datetime.now() - datetime.timedelta(hours=i)
                temp = np.random.normal(75, 10)  # Temperature around 75°F
                pressure = np.random.normal(30, 5)  # Pressure around 30 PSI
                vibration = np.random.normal(2.5, 0.5)  # Vibration around 2.5 mm/s
                power = np.random.normal(15, 3)  # Power around 15 kW
                
                cursor.execute('''
                    INSERT OR REPLACE INTO sensor_data 
                    (equipment_id, timestamp, temperature, pressure, vibration, power_consumption)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (eq_id, timestamp, temp, pressure, vibration, power))
        
        conn.commit()
        conn.close()
    
    def get_equipment_info(self, equipment_id: int) -> Dict:
        """Get equipment information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM equipment WHERE id = ?', (equipment_id,))
        result = cursor.fetchone()
        
        if result:
            columns = [desc[0] for desc in cursor.description]
            equipment_info = dict(zip(columns, result))
        else:
            equipment_info = {}
            
        conn.close()
        return equipment_info
    
    def get_sensor_data(self, equipment_id: int, hours: int = 24) -> pd.DataFrame:
        """Get recent sensor data"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM sensor_data 
            WHERE equipment_id = ? AND timestamp > datetime('now', '-{} hours')
            ORDER BY timestamp DESC
        '''.format(hours)
        
        df = pd.read_sql_query(query, conn, params=(equipment_id,))
        conn.close()
        return df
    
    def get_maintenance_history(self, equipment_id: int) -> pd.DataFrame:
        """Get maintenance history"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM maintenance_logs 
            WHERE equipment_id = ?
            ORDER BY date DESC
        '''
        
        df = pd.read_sql_query(query, conn, params=(equipment_id,))
        conn.close()
        return df

class KnowledgeBase:
    """Manages technical documentation and procedures"""
    
    def __init__(self, openai_api_key: str):
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.vector_store = None
        self.load_knowledge_base()
    
    def load_knowledge_base(self):
        """Load and process technical documentation"""
        # Sample technical documentation
        docs = [
            "Centrifugal Pump Troubleshooting: If pump is not priming, check suction line for air leaks. Verify impeller is not clogged. Check shaft seal for wear.",
            "Electric Motor Diagnostics: High vibration indicates bearing wear or misalignment. Temperature above 80°C suggests overloading or cooling issues.",
            "Air Compressor Maintenance: Regular oil changes every 2000 hours. Check belt tension monthly. Replace air filter every 6 months.",
            "Pump Cavitation Signs: Unusual noise, reduced flow, increased vibration. Solution: Check NPSH, reduce suction lift, increase suction pipe diameter.",
            "Motor Overheating Causes: Overloading, poor ventilation, dirty cooling fins, bearing failure. Check current draw and compare to nameplate.",
            "Compressor Oil Analysis: Dark oil indicates contamination. Metal particles suggest component wear. Change oil immediately if contaminated.",
            "Vibration Analysis: 1x RPM indicates unbalance. 2x RPM suggests misalignment. High frequency indicates bearing issues.",
            "Pressure Drop Troubleshooting: Check for clogged filters, closed valves, pipe restrictions. Calculate system curve vs pump curve.",
            "Electrical Safety Procedures: Always lock out power before maintenance. Use appropriate PPE. Test circuits with multimeter before work.",
            "Preventive Maintenance Schedule: Pumps - monthly inspection, quarterly alignment check. Motors - annual bearing lubrication, thermal scan."
        ]
        
        # Create embeddings
        texts = docs
        metadatas = [{"source": f"manual_page_{i}"} for i in range(len(docs))]
        
        self.vector_store = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
    
    def search_knowledge(self, query: str, k: int = 3) -> List[str]:
        """Search knowledge base for relevant information"""
        if self.vector_store:
            results = self.vector_store.similarity_search(query, k=k)
            return [doc.page_content for doc in results]
        return []

class ImageAnalyzer:
    """Analyzes equipment images using GPT-4V"""
    
    def __init__(self, openai_api_key: str):
        self.client = openai.OpenAI(api_key=openai_api_key)
    
    def analyze_image(self, image: Image.Image, context: str = "") -> str:
        """Analyze equipment image for issues"""
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        prompt = f"""
        As an expert field technician, analyze this equipment image for potential issues.
        
        Context: {context}
        
        Look for:
        - Visible damage, corrosion, or wear
        - Leaks or fluid stains
        - Misalignment or loose components
        - Unusual deposits or discoloration
        - Safety hazards
        
        Provide a detailed assessment with:
        1. Observed conditions
        2. Potential issues identified
        3. Recommended actions
        4. Safety considerations
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error analyzing image: {str(e)}"

class SensorDataAnalyzer(BaseTool):
    """LangChain tool for analyzing sensor data"""
    
    name = "sensor_data_analyzer"
    description = "Analyzes equipment sensor data to identify anomalies and trends"
    
    def __init__(self, db: EquipmentDatabase):
        super().__init__()
        self.db = db
    
    def _run(self, equipment_id: str, hours: str = "24") -> str:
        """Analyze sensor data for anomalies"""
        try:
            eq_id = int(equipment_id)
            hours_int = int(hours)
            
            df = self.db.get_sensor_data(eq_id, hours_int)
            
            if df.empty:
                return "No sensor data available for this equipment."
            
            # Calculate statistics
            stats = df[['temperature', 'pressure', 'vibration', 'power_consumption']].describe()
            
            # Identify anomalies (values beyond 2 standard deviations)
            anomalies = []
            for col in ['temperature', 'pressure', 'vibration', 'power_consumption']:
                mean_val = df[col].mean()
                std_val = df[col].std()
                high_threshold = mean_val + 2 * std_val
                low_threshold = mean_val - 2 * std_val
                
                high_anomalies = df[df[col] > high_threshold]
                low_anomalies = df[df[col] < low_threshold]
                
                if not high_anomalies.empty:
                    anomalies.append(f"High {col}: {len(high_anomalies)} readings above {high_threshold:.2f}")
                if not low_anomalies.empty:
                    anomalies.append(f"Low {col}: {len(low_anomalies)} readings below {low_threshold:.2f}")
            
            result = f"Sensor Data Analysis (Last {hours_int} hours):\n"
            result += f"Data points: {len(df)}\n"
            result += f"Temperature: {df['temperature'].mean():.1f}°F (±{df['temperature'].std():.1f})\n"
            result += f"Pressure: {df['pressure'].mean():.1f} PSI (±{df['pressure'].std():.1f})\n"
            result += f"Vibration: {df['vibration'].mean():.2f} mm/s (±{df['vibration'].std():.2f})\n"
            result += f"Power: {df['power_consumption'].mean():.1f} kW (±{df['power_consumption'].std():.1f})\n"
            
            if anomalies:
                result += "\nAnomalies detected:\n" + "\n".join(anomalies)
            else:
                result += "\nNo significant anomalies detected."
            
            return result
            
        except Exception as e:
            return f"Error analyzing sensor data: {str(e)}"
    
    async def _arun(self, equipment_id: str, hours: str = "24") -> str:
        return self._run(equipment_id, hours)

class MaintenanceHistoryTool(BaseTool):
    """LangChain tool for retrieving maintenance history"""
    
    name = "maintenance_history"
    description = "Retrieves maintenance history and previous issues for equipment"
    
    def __init__(self, db: EquipmentDatabase):
        super().__init__()
        self.db = db
    
    def _run(self, equipment_id: str) -> str:
        """Get maintenance history for equipment"""
        try:
            eq_id = int(equipment_id)
            df = self.db.get_maintenance_history(eq_id)
            
            if df.empty:
                return "No maintenance history available for this equipment."
            
            result = f"Maintenance History (Last {len(df)} records):\n"
            for _, row in df.head(5).iterrows():  # Show last 5 records
                result += f"\nDate: {row['date']}\n"
                result += f"Technician: {row['technician']}\n"
                result += f"Issue: {row['issue_description']}\n"
                result += f"Resolution: {row['resolution']}\n"
                if row['parts_used']:
                    result += f"Parts Used: {row['parts_used']}\n"
                result += "-" * 40 + "\n"
            
            return result
            
        except Exception as e:
            return f"Error retrieving maintenance history: {str(e)}"
    
    async def _arun(self, equipment_id: str) -> str:
        return self._run(equipment_id)

class KnowledgeSearchTool(BaseTool):
    """LangChain tool for searching technical knowledge base"""
    
    name = "knowledge_search"
    description = "Searches technical documentation and procedures for relevant information"
    
    def __init__(self, kb: KnowledgeBase):
        super().__init__()
        self.kb = kb
    
    def _run(self, query: str) -> str:
        """Search knowledge base"""
        try:
            results = self.kb.search_knowledge(query)
            if results:
                return "Relevant technical information found:\n\n" + "\n\n".join(results)
            else:
                return "No relevant technical information found."
        except Exception as e:
            return f"Error searching knowledge base: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        return self._run(query)

class FieldTechnicianAgent:
    """Main agent orchestrating all tools and capabilities"""
    
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.db = EquipmentDatabase()
        self.kb = KnowledgeBase(openai_api_key)
        self.image_analyzer = ImageAnalyzer(openai_api_key)
        
        # Initialize tools
        self.tools = [
            SensorDataAnalyzer(self.db),
            MaintenanceHistoryTool(self.db),
            KnowledgeSearchTool(self.kb)
        ]
        
        # Initialize LangChain agent
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            openai_api_key=openai_api_key
        )
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True
        )
    
    def diagnose_equipment(self, equipment_id: int, issue_description: str = "", 
                          image: Optional[Image.Image] = None) -> Dict[str, Any]:
        """Comprehensive equipment diagnosis"""
        
        # Get equipment info
        equipment_info = self.db.get_equipment_info(equipment_id)
        
        if not equipment_info:
            return {"error": "Equipment not found"}
        
        diagnosis_results = {
            "equipment_info": equipment_info,
            "sensor_analysis": None,
            "image_analysis": None,
            "recommendations": None,
            "maintenance_history": None
        }
        
        # Build context for AI analysis
        context = f"""
        Equipment: {equipment_info['name']} ({equipment_info['type']})
        Model: {equipment_info['model']}
        Location: {equipment_info['location']}
        Issue Description: {issue_description}
        """
        
        # Analyze image if provided
        if image:
            diagnosis_results["image_analysis"] = self.image_analyzer.analyze_image(image, context)
        
        # Use agent to analyze all available data
        agent_query = f"""
        I need to diagnose an issue with equipment ID {equipment_id}.
        
        Equipment Details:
        - Name: {equipment_info['name']}
        - Type: {equipment_info['type']}
        - Model: {equipment_info['model']}
        - Location: {equipment_info['location']}
        
        Issue Description: {issue_description}
        
        Please:
        1. Analyze the current sensor data for anomalies
        2. Check the maintenance history for similar issues
        3. Search the knowledge base for relevant troubleshooting information
        4. Provide specific recommendations for diagnosis and repair
        
        Focus on safety considerations and step-by-step procedures.
        """
        
        try:
            agent_response = self.agent.run(agent_query)
            diagnosis_results["recommendations"] = agent_response
        except Exception as e:
            diagnosis_results["recommendations"] = f"Agent analysis failed: {str(e)}"
        
        return diagnosis_results

def main():
    """Main Streamlit application"""
    
    st.title("🔧 AI Field Technician Assistant")
    st.markdown("**Multi-Modal Equipment Diagnosis & Maintenance Support**")
    
    # Sidebar for API key and configuration
    with st.sidebar:
        st.header("Configuration")
        
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key to enable AI features"
        )
        
        if not openai_api_key:
            st.warning("Please enter your OpenAI API key to continue")
            st.stop()
        
        st.header("Equipment Selection")
        equipment_options = {
            1: "Pump A-101 (Centrifugal Pump)",
            2: "Motor B-205 (Electric Motor)", 
            3: "Compressor C-301 (Air Compressor)"
        }
        
        selected_equipment = st.selectbox(
            "Select Equipment",
            options=list(equipment_options.keys()),
            format_func=lambda x: equipment_options[x]
        )
    
    # Initialize agent
    if 'agent' not in st.session_state:
        st.session_state.agent = FieldTechnicianAgent(openai_api_key)
    
    # Main interface tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Diagnosis", "📊 Sensor Data", "📋 Maintenance History", "📚 Knowledge Base"])
    
    with tab1:
        st.header("Equipment Diagnosis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            issue_description = st.text_area(
                "Describe the Issue",
                placeholder="e.g., Unusual noise from pump, high vibration, overheating...",
                height=100
            )
            
            uploaded_image = st.file_uploader(
                "Upload Equipment Photo (Optional)",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a photo of the equipment for visual analysis"
            )
            
            if st.button("🔍 Start Diagnosis", type="primary"):
                if issue_description.strip():
                    with st.spinner("Analyzing equipment..."):
                        
                        # Process uploaded image
                        image = None
                        if uploaded_image:
                            image = Image.open(uploaded_image)
                        
                        # Run diagnosis
                        results = st.session_state.agent.diagnose_equipment(
                            selected_equipment,
                            issue_description,
                            image
                        )
                        
                        if "error" in results:
                            st.error(results["error"])
                        else:
                            st.session_state.diagnosis_results = results
                else:
                    st.warning("Please describe the issue before starting diagnosis.")
        
        with col2:
            if uploaded_image:
                st.image(uploaded_image, caption="Equipment Photo", use_column_width=True)
        
        # Display diagnosis results
        if 'diagnosis_results' in st.session_state:
            results = st.session_state.diagnosis_results
            
            st.subheader("📋 Diagnosis Results")
            
            # Equipment Info
            with st.expander("Equipment Information", expanded=True):
                info = results["equipment_info"]
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Name:** {info['name']}")
                    st.write(f"**Type:** {info['type']}")
                    st.write(f"**Model:** {info['model']}")
                with col2:
                    st.write(f"**Location:** {info['location']}")
                    st.write(f"**Install Date:** {info['install_date']}")
                    st.write(f"**Last Maintenance:** {info['last_maintenance']}")
            
            # Image Analysis
            if results["image_analysis"]:
                with st.expander("Visual Analysis", expanded=True):
                    st.write(results["image_analysis"])
            
            # AI Recommendations
            if results["recommendations"]:
                with st.expander("AI Recommendations", expanded=True):
                    st.write(results["recommendations"])
    
    with tab2:
        st.header("📊 Real-Time Sensor Data")
        
        # Get sensor data
        db = st.session_state.agent.db
        df = db.get_sensor_data(selected_equipment, 24)
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Create plots
            fig = go.Figure()
            
            metrics = ['temperature', 'pressure', 'vibration', 'power_consumption']
            colors = ['red', 'blue', 'green', 'orange']
            
            for metric, color in zip(metrics, colors):
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df[metric],
                    mode='lines',
                    name=metric.replace('_', ' ').title(),
                    line=dict(color=color)
                ))
            
            fig.update_layout(
                title="Sensor Data Trends (Last 24 Hours)",
                xaxis_title="Time",
                yaxis_title="Values",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Current readings
            st.subheader("Current Readings")
            latest = df.iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Temperature", f"{latest['temperature']:.1f}°F")
            with col2:
                st.metric("Pressure", f"{latest['pressure']:.1f} PSI")
            with col3:
                st.metric("Vibration", f"{latest['vibration']:.2f} mm/s")
            with col4:
                st.metric("Power", f"{latest['power_consumption']:.1f} kW")
            
        else:
            st.info("No sensor data available for this equipment.")
    
    with tab3:
        st.header("📋 Maintenance History")
        
        # Get maintenance history
        db = st.session_state.agent.db
        maintenance_df = db.get_maintenance_history(selected_equipment)
        
        if not maintenance_df.empty:
            for _, record in maintenance_df.iterrows():
                with st.expander(f"{record['date']} - {record['issue_description'][:50]}..."):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Date:** {record['date']}")
                        st.write(f"**Technician:** {record['technician']}")
                    with col2:
                        if record['parts_used']:
                            st.write(f"**Parts Used:** {record['parts_used']}")
                    
                    st.write(f"**Issue:** {record['issue_description']}")
                    st.write(f"**Resolution:** {record['resolution']}")
        else:
            st.info("No maintenance history available for this equipment.")
    
    with tab4:
        st.header("📚 Technical Knowledge Base")
        
        search_query = st.text_input(
            "Search Technical Documentation",
            placeholder="e.g., pump cavitation, motor overheating, vibration analysis..."
        )
        
        if search_query:
            kb = st.session_state.agent.kb
            results = kb.search_knowledge(search_query)
            
            if results:
                st.subheader("Relevant Information:")
                for i, result in enumerate(results, 1):
                    with st.expander(f"Result {i}"):
                        st.write(result)
            else:
                st.info("No relevant information found. Try different search terms.")

if __name__ == "__main__":
    main()