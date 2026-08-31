from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from datetime import datetime
import os

def generate_dossier(target_name, risk_score, tox_results_dict, output_filename):
    c = canvas.Canvas(output_filename, pagesize=letter)
    width, height = letter
    
    # --- PAGE 1: TITLE PAGE ---
    c.setFont("Helvetica-Bold", 24)
    c.setFillColor(colors.HexColor("#3730A3"))
    c.drawString(50, height - 100, "StartNerve")
    
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.HexColor("#111827"))
    c.drawString(50, height - 130, "STARTNERVE INTELLIGENCE | PRE-MANUFACTURING AUDIT")
    
    c.setLineWidth(2)
    c.setStrokeColor(colors.HexColor("#4F46E5"))
    c.line(50, height - 140, width - 50, height - 140)
    
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 200, "REGULATORY RISK &")
    c.drawString(50, height - 225, "IMPURITY COMPLIANCE DOSSIER")
    
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 280, f"Target Active Pharmaceutical Ingredient (API): {target_name}")
    c.drawString(50, height - 300, f"Audit Date: {datetime.now().strftime('%Y-%m-%d')}")
    
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(colors.red)
    c.drawString(50, height - 350, "STRICTLY CONFIDENTIAL")
    c.setFont("Helvetica", 10)
    c.setFillColor(colors.black)
    c.drawString(50, height - 365, "This document contains proprietary in-silico intelligence.")
    
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, 50, "Page 1 | CONFIDENTIAL & PROPRIETARY - StartNerve Intelligence")
    c.showPage()
    
    # --- PAGE 2: EXECUTIVE SUMMARY & STRUCTURE ---
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 50, "1.0 Executive Summary & Risk Assessment")
    
    c.setFont("Helvetica", 10)
    summary_text = [
        f"This intelligence dossier provides a comprehensive in-silico compliance and toxicity risk assessment for the synthesis",
        f"and manufacturing profile of {target_name}. Utilizing StartNerve's proprietary Hybrid Bio-Engine,",
        "this audit screens for structural alerts related to mutagenicity, carcinogenicity, and off-target nuclear",
        "receptor binding. The primary objective is to proactively flag potential ICH M7 mutagenic impurities prior to",
        "wet-lab allocation, DMF filing, and commercial manufacturing, thereby mitigating the risk of FDA 483 observations or",
        "batch rejections."
    ]
    
    y_pos = height - 80
    for line in summary_text:
        c.drawString(50, y_pos, line)
        y_pos -= 15
        
    # Draw Molecule Image
    if os.path.exists("molecule.png"):
        c.drawImage("molecule.png", 50, y_pos - 200, width=200, height=200, preserveAspectRatio=True)
    
    # Draw Risk Score Box
    box_y = y_pos - 150
    c.setFillColor(colors.HexColor("#F3F4F6"))
    c.rect(300, box_y, 200, 100, fill=1, stroke=0)
    
    if risk_score > 50:
        c.setFillColor(colors.HexColor("#991B1B")) # Red
    else:
        c.setFillColor(colors.HexColor("#065F46")) # Green
        
    c.setFont("Helvetica-Bold", 36)
    c.drawCentredString(400, box_y + 45, f"{risk_score}/100")
    
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(colors.black)
    c.drawCentredString(400, box_y + 25, "COMPLIANCE RISK")
    
    # Section 2.0
    y_pos = box_y - 60
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_pos, "2.0 Regulatory Context: ICH M7 Guidelines")
    
    c.setFont("Helvetica", 10)
    ich_text = [
        "The International Council for Harmonisation (ICH) M7 guideline outlines the assessment and control of DNA reactive",
        "(mutagenic) impurities in pharmaceuticals to limit potential carcinogenic risk. Regulatory agencies globally (FDA, EMA,",
        "CDSCO) strictly enforce these limits. The StartNerve Bio-Engine evaluates the proposed molecular structure against",
        "known mutagenic pharmacophores to assign a predictive risk score, allowing manufacturers to alter synthesis routes or",
        "establish appropriate HPLC/GC analytical control strategies early in the product lifecycle."
    ]
    
    y_pos -= 30
    for line in ich_text:
        c.drawString(50, y_pos, line)
        y_pos -= 15
        
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, 50, "Page 2 | CONFIDENTIAL & PROPRIETARY - StartNerve Intelligence")
    c.showPage()
    
    # --- PAGE 3: HYBRID AI RESULTS & RECOMMENDATIONS ---
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 50, "3.0 In-Silico Toxicity & Mutagenicity Scan Results")
    
    y_pos = height - 90
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, y_pos, "BIOLOGICAL ENDPOINT / PATHWAY")
    c.drawString(400, y_pos, "PREDICTED OUTCOME")
    
    c.setLineWidth(1)
    c.line(50, y_pos - 5, width - 50, y_pos - 5)
    
    y_pos -= 25
    c.setFont("Helvetica", 10)
    
    # Loop through all 13 endpoints from the Streamlit App
    for endpoint, result in tox_results_dict.items():
        c.setFillColor(colors.black)
        c.drawString(50, y_pos, endpoint)
        
        if "FAIL" in result:
            c.setFillColor(colors.HexColor("#EF4444")) # Red
            c.setFont("Helvetica-Bold", 10)
        else:
            c.setFillColor(colors.HexColor("#10B981")) # Green
            c.setFont("Helvetica-Bold", 10)
            
        c.drawString(400, y_pos, result)
        c.setFont("Helvetica", 10)
        y_pos -= 20
        
    # Section 4.0
    y_pos -= 40
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_pos, "4.0 Strategic Recommendations & Next Steps")
    
    y_pos -= 30
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, y_pos, "1. ANALYTICAL MONITORING: ")
    c.setFont("Helvetica", 10)
    c.drawString(200, y_pos, "Based on the flagged stress response pathways, we strongly recommend")
    y_pos -= 15
    c.drawString(50, y_pos, "establishing sensitive LC-MS/MS methods to monitor for related degradation products and intermediates.")
    
    y_pos -= 25
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, y_pos, "2. VENDOR AUDIT: ")
    c.setFont("Helvetica", 10)
    c.drawString(150, y_pos, "If this API is being sourced from a third-party Contract Manufacturing Organization (CMO),")
    y_pos -= 15
    c.drawString(50, y_pos, "request their DMF (Drug Master File) open part to verify their specific synthesis route against this risk profile.")
    
    y_pos -= 50
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_pos, "Bhavesh Satbhai")
    c.setFont("Helvetica", 10)
    c.drawString(50, y_pos - 15, "Lead Architect & Founder, StartNerve Intelligence")
    
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, 50, "Page 3 | CONFIDENTIAL & PROPRIETARY - StartNerve Intelligence")
    
    c.save()