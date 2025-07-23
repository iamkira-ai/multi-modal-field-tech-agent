import base64
import io
from PIL import Image
import streamlit as st
from typing import Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np

def encode_image(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def resize_image(image: Image.Image, max_size: Tuple[int, int] = (1024, 1024)) -> Image.Image:
    """Resize image while maintaining aspect ratio"""
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image

def validate_image(uploaded_file, max_size_mb: int = 10) -> Optional[Image.Image]:
    """Validate and process uploaded image"""
    if uploaded_file is None:
        return None
    
    # Check file size
    if uploaded_file.size > max_size_mb * 1024 * 1024:
        st.error(f"Image too large. Maximum size is {max_size_mb}MB")
        return None
    
    try:
        image = Image.open(uploaded_file)
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        return resize_image(image)
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def generate_anomaly_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive anomaly analysis report"""
    if df.empty:
        return {"error": "No data provided"}
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    report = {
        "summary": {},
        "anomalies": {},
        "trends": {},
        "recommendations": []
    }
    
    for col in numeric_columns:
        if col in df.columns:
            data = df[col].dropna()
            if len(data) > 0:
                mean_val = data.mean()
                std_val = data.std()
                
                # Statistical summary
                report["summary"][col] = {
                    "mean": round(mean_val, 2),
                    "std": round(std_val, 2),
                    "min": round(data.min(), 2),
                    "max": round(data.max(), 2),
                    "latest": round(data.iloc[-1], 2) if len(data) > 0 else None
                }
                
                # Anomaly detection (2-sigma rule)
                threshold_high = mean_val + 2 * std_val
                threshold_low = mean_val - 2 * std_val
                
                anomalies = data[(data > threshold_high) | (data < threshold_low)]
                if len(anomalies) > 0:
                    report["anomalies"][col] = {
                        "count": len(anomalies),
                        "percentage": round(len(anomalies) / len(data) * 100, 2),
                        "severity": "High" if len(anomalies) / len(data) > 0.1 else "Medium"
                    }
                
                # Trend analysis (simple linear regression on last 20 points)
                if len(data) >= 20:
                    recent_data = data.tail(20)
                    x = np.arange(len(recent_data))
                    slope = np.polyfit(x, recent_data, 1)[0]
                    report["trends"][col] = {
                        "slope": round(slope, 4),
                        "direction": "Increasing" if slope > 0.01 else "Decreasing" if slope < -0.01 else "Stable"
                    }
    
    # Generate recommendations based on analysis
    recommendations = []
    for col, anomaly_info in report["anomalies"].items():
        if anomaly_info["severity"] == "High":
            recommendations.append(f"⚠️ High anomaly rate in {col} ({anomaly_info['percentage']}%). Immediate inspection recommended.")
    
    for col, trend_info in report["trends"].items():
        if trend_info["direction"] in ["Increasing", "Decreasing"]:
            recommendations.append(f"📈 {col} showing {trend_info['direction'].lower()} trend. Monitor closely.")
    
    report["recommendations"] = recommendations
    return report