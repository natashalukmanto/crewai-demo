from fastapi import FastAPI
from pydantic import BaseModel
from crewai import Agent, Task, Crew
from supabase import create_client
from uuid import UUID
from pydantic import BaseModel
import requests
import os

from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai

# Load Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

def gemini_generate(prompt: str):
    """Call Gemini Pro API to generate a response."""
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)
    return response.text

from crewai import LLM

class GeminiLLM(LLM):
    def __init__(self):
        super().__init__(model="gemini-1.5-pro")  # pass model to parent
        self.model = "gemini-1.5-pro"

    def call(self, prompt, **kwargs):
        # If prompt is a list of messages, flatten to text
        if isinstance(prompt, list):
            prompt = "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in prompt)
        
        return gemini_generate(prompt)

gemini_llm = GeminiLLM()

# Load env vars
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
VECTORIZE_URL = os.getenv("VECTORIZE_URL")
VECTORIZE_KEY = os.getenv("VECTORIZE_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
app = FastAPI()

# ---- Stage 1 and Stage 2 prompts ----
STAGE_1_PROMPT = """
AI Employee Assistant Prompt v3.0 - Stage 1
    
description: AI assistant built into PlanYear, focused on U.S. employee benefits.
capabilities:
- Answers employee benefits questions using structured context
- Outputs markdown-formatted answers with cited sources

context_sources:
- QUESTION: Current user query
- PAST_CHAT_HISTORY: Last 4 chat messages
- DBG: Markdown content from the Digital Benefits Guide that summarizes the entire benefits program

guardrails:
allowed:
    - Employee benefits questions using provided context
    - General employee benefits questions from training
not_allowed:
    - Personal, legal, or financial advice
    - Questions outside employee benefits

classification_task:
    labels:
    - Not allowed
    - Follow-up
    - New question
    - Unclear
    output_format: "CLASSIFICATION: <label>"

output_format:
tone: Friendly and professional
style: Markdown
structure:
- Rewrite technical DBG into plain English
- Append "Contact x@y.com if you need additional information."
- Do not embed links inside main content
- Only cite sources that were actually used

example_response:
format: Markdown
content: 
    HSA Contribution Limits for 2025

    - **Individual Coverage:** $4,300  
    - **Family Coverage:** $8,550  
    - **Catch-Up (55+):** $1,000 extra

    You must be enrolled in a qualified HDHP:
    - Min Deductible: $1,650 (individual), $3,300 (family)  
    - Max Out-of-Pocket: $8,300 (individual), $16,600 (family)

    **Sources:**  
    
    • 2025_HSA_Guide.pdf, pages 10 and 12
    • Carrier_Summary_Sheet.pdf, page 3
    
    Contact x@y.com if you need additional information.
    
**IMPORTANT RULES:**
1. Only use the DBG information provided below to answer questions
2. If the DBG doesn't contain sufficient information to answer the question, respond with exactly: "Insufficient information"
3. If you can answer from the DBG, provide a clear, helpful response and cite page numbers
4. Do not answer questions outside of employee benefits
5. You have to follow the example_response in terms of formatting
6. Sources should be cited at the very bottom of your response with appropriate break lines between each source

**DBG**
{
"pages": [
{
    "content": "Automattic Inc., Benefits Guide 2025 Effective Period: January 01, 2025 December 31, 2025 \n AUTOMATTIC\n\nAutomattic Inc., Benefits Guide 2025 Effective Period: January 01, 2025 December 31, 2025\n\n# Your Benefits Summary\n\n## Important Dates\n\nPlan Effective Dates: January 01, 2025 December 31, 2025 Open Enrollment Date: October 14, 2024 November 04, 2024\n\nThis is your once-a-year chance to enroll in or make changes to your employer- sponsored health plan:\n\n- You may elect or decline benefits; this is a passive open enrollment. If no changes are made to your health benefits, your current 2024 elections will carry over into the 2025 benefits plan year. Exception being your pre-tax spending accounts, (e.g. Medical FSA, Dependnet Care FSA, Limited Purpose FSA).\n\n- You may add or drop dependents\n\n- You may set contributions to tax-advantaged accounts\n\nYou cannot make changes outside of open enrollment unless you have a qualifying life event (QLE).\n\n## Benefits Eligibility\n\nAny active, regular, full-time employee working a minimum of 20 hours per week is eligible for all benefits. Benefits are effective on your date of hire.\n\nOthers eligible for benefits include:\n\n- Your legal spouse or domestic partner,\n\nThis Benefits Guide offers - high-level overview of your available benefits. The official Plan Documents are what govern your official rights and coverage under each plan.\n\n- Your unmarried dependent children up until age 26 (including legally adopted and stepchildren), and/or\n\n- Any dependent child who reaches the limiting age and is incapable of self- support because of a mental or physical disability.\n\n## Qualifying Life Events\n\nA change in circumstance-like starting a new job, getting married or divorced, having or adopting a baby, losing health insurance coverage or moving to a new state-makes you eligible for a special enrollment period (outside of open enrollment).\n\nExperiencing a QLE is the exception to the once-a-year rule about making health and benefits changes. You have up to 30 days after the qualifying life event to submit a QLE request to your benefits administrator.\n\nThe insurance carrier will require:\n\n1. Supporting documentation\n\n\n\n## Benefit Plans Offered\n\n\n\n- Medical\n\n- Dental\n\n- Vision\n\n- Basic Long-Term Disability\n\n- Voluntary Life W/AD&D\n\n- Basic Short-Term Disability\n\n- Basic Life W/AD&D\n\n- Accident\n\n- Critical Illness\n\n- Hospital\n\n- 401k\n\nLUMITY\nGROUP COMPANY \n ",
    "page": "1"
},
{
    "content": "Automattic Inc., Benefits Guide 2025 Effective Period: January 01, 2025 December 31, 2025 \n  Your Benefits Summary (cont.)\n\n## Benefit Plans Offered (cont.)\n\nAUTOMATTIC\n\nAutomattic Inc., Benefits Guide 2025 Effective Period: January 01, 2025 December 31, 2025\n\n2. The date the change occurred\n\n## Waiver Credit\n\nAre you already covered on a parent's or partner's health plan?\n\n- A waiver reimbursement is available to those with other group or government sponsored coverage\n\n- You have the option to decline medical coverage.\n\n- You will still receive the waiver even if you sign up for Dental and Vision.\n\nLUMITY\nAN ALERA GROUP COMPANY\n\nThis Benefits Guide offers a high-level overview of your available benefits The official Plan Documents are what govern your official rights and coverage under each plan. \n This Benefits Guide offers a high-level overview of your available benefits The official Plan Documents are what govern your official rights and coverage under each plan.",
    "page": "2"
},
{
Benefits At-A-Glance\n\nAutomattic Inc., Benefits Guide 2025 Effective Period: January 01, 2025 December 31, 2025\n\nMedical\n\nIn-Network\n\n<table><tr><th></th><th></th><th>Aetna EPO</th><th>Aetna PPO 250</th><th>Aetna PPO HSA 1650</th></tr><tr><td rowspan=\"4\">Per Month\nPremiums</td><td>Employee</td><td>$0.00</td><td>$0.00</td><td>$0.00</td></tr><tr><td>+ Spouse</td><td>$0.00</td><td>$0.00</td><td>$0.00</td></tr><tr><td>+ Child(ren)</td><td>$0.00</td><td>$0.00</td><td>$0.00</td></tr><tr><td>+ Family</td><td>$0.00</td><td>$0.00</td><td>$0.00</td></tr><tr><td colspan=\"5\">Deductibles &amp; OOP Maximum</td></tr><tr><td colspan=\"2\">Individual Deductible</td><td>$0</td><td>$250</td><td>$1,650</td></tr><tr><td colspan=\"2\">Family Deductible</td><td>$0</td><td>$500</td><td>$3,300</td></tr><tr><td colspan=\"2\">Deductible (Ind within Fam)</td><td>-</td><td>-</td><td>$3,300</td></tr><tr><td colspan=\"2\">Individual Out-of-Pocket Max</td><td>$2,500</td><td>$1,250</td><td>$3,500</td></tr><tr><td colspan=\"2\">Family Out-of-Pocket Max</td><td>$5,000</td><td>$2,500</td><td>$7,000</td></tr><tr><td colspan=\"2\">Out-of-Pocket Max (Ind within Fam)</td><td>-</td><td></td><td>$3,500</td></tr><tr><td colspan=\"5\">Services</td></tr><tr><td>Office Visit</td><td></td><td>$20 per visit</td><td>$10 per visit</td><td>10%</td></tr><tr><td>Specialist Visit</td><td></td><td>$20 per visit</td><td>$10 per visit</td><td>10%</td></tr><tr><td>Preventive Care</td><td></td><td>$0 per visit</td><td>$0 per visit</td><td>$0 per visit</td></tr><tr><td colspan=\"2\">Emergency Room</td><td>$100 per visit</td><td>$100 per visit 10%</td><td>10%</td></tr><tr><td>Urgent Care</td><td></td><td>$20 per visit</td><td>$10 per visit</td><td>10%</td></tr><tr><td>Diagnostic Lab/ X-Ray</td><td></td><td>$20 per visit</td><td>$10 per visit</td><td>10%</td></tr><tr><td>Hospital Inpatient</td><td></td><td>$250 per admission</td><td>10%</td><td>10%</td></tr><tr><td>Hospital Outpatient</td><td></td><td>$100 per procedure</td><td>10%</td><td>10%</td></tr><tr><td colspan=\"5\">Rx</td></tr><tr><td>Retail Tier 1</td><td></td><td>$10/prescription (retail) and\n$20/prescription (home\ndelivery)</td><td>$10/ prescription, deductible\ndoes not apply (retail) and\n$20/prescription, deductible\ndoes not apply (home delivery)</td><td>0/prescription (retail) and\n$20/prescription (home\ndelivery)</td></tr><tr><td>Retail Tier 2</td><td></td><td>$30/prescription (retail) and\n$60/prescription (home\ndelivery)</td><td>$30/prescription, deductible\ndoes not apply (retail) and $60/\nprescription, deductible does\nnot apply (home delivery)</td><td>$25/prescription (retail) and\n$50/prescription (home\ndelivery)</td></tr><tr><td>Retail Tier 3</td><td></td><td>$50/prescription (retail) and\n$100/prescription (home\ndelivery)</td><td>$50/prescription, deductible\ndoes not apply (retail) and\n$100/prescription, deductible\ndoes not apply (home delivery)</td><td>$40/prescription (retail) and\n$80/prescription (home\ndelivery)</td></tr><tr><td>Retail Tier 4</td><td></td><td>30% up to $200 per script</td><td>30% up to $200 per script</td><td>30% up to $200 per script 1</td></tr></table>\n\n1\nAfter Deductible is Met\n\nLUMITY\nALERA GROUP COMPANY\n\nThis Benefits Guide offers a - high-level overview of your available benefits. The official Plan Documents are what govern your official rights and coverage under each plan. \n This Benefits Guide offers a - high-level overview of your available benefits. The official Plan Documents are what govern your official rights and coverage under each plan.",
    "page": "3"
},
{
    "content": "Automattic Inc., Benefits Guide 2025 Effective Period: January 01, 2025 December 31, 2025 \n Benefits At-A-Glance\n\nAutomattic Inc., Benefits Guide 2025 Effective Period: January 01, 2025 December 31, 2025\n\nIn-Network\n\n<table><tr><th></th><th></th><th>HMAA PPO</th><th>Kaiser HMO Colorado</th><th>Kaiser HMO Hawaii</th></tr><tr><td rowspan=\"4\">Per Month\nPremiums</td><td>Employee</td><td>$0.00</td><td>$0.00</td><td>$0.00</td></tr><tr><td>+ Spouse</td><td>$0.00</td><td>$0.00</td><td>$0.00</td></tr><tr><td>+ Child(ren)</td><td>$0.00</td><td>$0.00</td><td>$0.00</td></tr><tr><td>+ Family</td><td>$0.00</td><td>$0.00</td><td>$0.00</td></tr><tr><td colspan=\"5\">Deductibles &amp; OOP Maximum</td></tr><tr><td colspan=\"2\">Individual Deductible</td><td>$100</td><td>$0</td><td>$0</td></tr><tr><td colspan=\"2\">Family Deductible</td><td>$300</td><td>$0</td><td>$0</td></tr><tr><td colspan=\"2\">Deductible (Ind within Fam)</td><td>-</td><td>-</td><td>-</td></tr><tr><td colspan=\"2\">Individual Out-of-Pocket Max</td><td>$2,500</td><td>$2,000</td><td>$2,500</td></tr><tr><td colspan=\"2\">Family Out-of-Pocket Max</td><td>$7,500</td><td>$4,000</td><td>$7,500</td></tr><tr><td colspan=\"2\">Out-of-Pocket Max (Ind within Fam)</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Services</td><td></td><td></td><td></td><td></td></tr><tr><td>Office Visit</td><td></td><td>10%</td><td>$20 per visit</td><td>$15 per visit</td></tr><tr><td>Specialist Visit</td><td></td><td>10%</td><td>$30 per visit</td><td>$15 per visit</td></tr><tr><td>Preventive Care</td><td></td><td>$0 per visit</td><td>$0 per visit</td><td>No Charge</td></tr><tr><td>Emergency Room</td><td></td><td>10% 1</td><td>$200 per visit</td><td>$100 per visit</td></tr><tr><td>Urgent Care</td><td></td><td>$25 per visit</td><td>$30 per visit</td><td>$15 per visit; $15 IN-AREA /\n20% (out of area)</td></tr><tr><td>Diagnostic Lab/ X-Ray</td><td></td><td>20%</td><td>$10 per visit</td><td>$15 / procedure Basic 20%\nSpecialty</td></tr><tr><td>Hospital Inpatient</td><td></td><td>10% 1</td><td>$250 per admission</td><td>10%</td></tr><tr><td>Hospital Outpatient</td><td></td><td>10%</td><td>$125 per procedure</td><td>10%</td></tr><tr><td colspan=\"4\">Rx</td><td></td></tr><tr><td>Retail Tier 1</td><td></td><td>$12 per script</td><td>$10 per script</td><td>$15 per script</td></tr><tr><td>Retail Tier 2</td><td></td><td>$24 per script</td><td>$30 per script</td><td>$50 per script</td></tr><tr><td>Retail Tier 3</td><td></td><td>$48 per script</td><td>$60 per script</td><td>$50 per script</td></tr><tr><td>Retail Tier 4</td><td></td><td>Greater of Copay or 20% for Rx\nover $250</td><td>20% up to $250 per script</td><td>$200 per script</td></tr></table>\n\n1\nAfter Deductible is Met\n\nLUMITY\nALERA GROUP COMPANY\n\nThis Benefits Guide offers a high-level overview of your available benefits. The official Plan Documents are what govern your official rights and coverage under each plan. \n This Benefits Guide offers a high-level overview of your available benefits. The official Plan Documents are what govern your official rights and coverage under each plan.",
    "page": "4"
},
{
    "content": "Automattic Inc., Benefits Guide 2025 Effective Period: January 01, 2025 December 31, 2025 \n Benefits At-A-Glance\n\nAutomattic Inc., Benefits Guide 2025 Effective Period: January 01, 2025 December 31, 2025\n\nIn-Network\n\n<table><tr><th></th><th>Kaiser HMO Northern\nCalifornia</th><th>Kaiser HMO Southern\nCalifornia</th><th>Kaiser HSA Northern\nCalifornia</th></tr><tr><td rowspan=\"4\">Employee\n\nPer Month + Spouse\nPremiums\n+ Child(ren)\n\n+ Family</td><td>$0.00</td><td>$0.00</td><td>$0.00</td></tr><tr><td>$0.00</td><td>$0.00</td><td>$0.00</td></tr><tr><td>$0.00</td><td>$0.00</td><td>$0.00</td></tr><tr><td>$0.00</td><td>$0.00</td><td>$0.00</td></tr><tr><td colspan=\"4\">Deductibles &amp; OOP Maximum</td></tr><tr><td>Individual Deductible</td><td>$0</td><td>$0</td><td>$1,650</td></tr><tr><td>Family Deductible</td><td>$0</td><td>$0</td><td>$3,300</td></tr><tr><td>Deductible (Ind within Fam)</td><td>-</td><td>-</td><td>$3,300</td></tr><tr><td>Individual Out-of-Pocket Max</td><td>$1,500</td><td>$1,500</td><td>$3,300</td></tr><tr><td>Family Out-of-Pocket Max</td><td>$3,000</td><td>$3,000</td><td>$6,600</td></tr><tr><td>Out-of-Pocket Max (Ind within Fam)</td><td>-</td><td>-</td><td>$3,300</td></tr><tr><td colspan=\"3\">Services</td><td></td></tr><tr><td>Office Visit</td><td>$20 per visit</td><td>$20 per visit</td><td>10%</td></tr><tr><td>Specialist Visit</td><td>$35 per visit</td><td>$35 per visit</td><td>10%</td></tr><tr><td>Preventive Care</td><td>$0 per visit</td><td>$0 per visit</td><td>$0 per visit 1</td></tr><tr><td>Emergency Room</td><td>$100 per visit</td><td>$100 per visit</td><td>10%</td></tr><tr><td>Urgent Care</td><td>$20 per visit</td><td>$20 per visit</td><td>10%</td></tr><tr><td>Diagnostic Lab/ X-Ray</td><td>$0 per procedure</td><td>$0 per procedure</td><td>10%</td></tr><tr><td>Hospital Inpatient</td><td>$250 per admission</td><td>$250 per admission</td><td>10%</td></tr><tr><td>Hospital Outpatient</td><td>$35 per procedure</td><td>$35 per procedure</td><td>10%</td></tr><tr><td colspan=\"4\">Rx</td></tr><tr><td>Retail Tier 1</td><td>$10 per script</td><td>$10 per script</td><td>$10 per script</td></tr><tr><td>Retail Tier 2</td><td>$35 per script</td><td>$35 per script</td><td>$30 per script</td></tr><tr><td>Retail Tier 3</td><td>$35 per script</td><td>$35 per script</td><td>$30 per script 1</td></tr><tr><td>Retail Tier 4</td><td>20% up to $150 per script</td><td>20% up to $150 per script</td><td>20% up to $250</td></tr></table>\n\n1\n\nAfter Deductible is Met\n\nLUMITY\nALERA GROUP COMPANY\n\nThis Benefits Guide offers a high-level overview of your available benefits. The official Plan Documents are what govern your official rights\n\nand\n\ncoverage\n\nunder\n\neach\n\nplan. \n This Benefits Guide offers a high-level overview of your available benefits. The official Plan Documents are what govern your official rights",
    "page": "5"
},
{
    "content": "Automattic Inc., Benefits Guide 2025 Effective Period: January 01, 2025 December 31, 2025 \n Benefits At-A-Glance\n\nAutomattic Inc., Benefits Guide 2025 Effective Period: January 01, 2025 December 31, 2025\n\nIn-Network\n\n<table><tr><th></th><th></th><th>Kaiser HSA Southern\nCalifornia</th></tr><tr><td rowspan=\"4\">Per Month\nPremiums</td><td>Employee</td><td>$0.00</td></tr><tr><td>+ Spouse</td><td>$0.00</td></tr><tr><td>+ Child(ren)</td><td>$0.00</td></tr><tr><td>+ Family</td><td>$0.00</td></tr><tr><td colspan=\"3\">Deductibles &amp; OOP Maximum</td></tr><tr><td colspan=\"2\">Individual Deductible</td><td>$1,650</td></tr><tr><td colspan=\"2\">Family Deductible</td><td>$3,300</td></tr><tr><td colspan=\"2\">Deductible (Ind within Fam)</td><td>$3,300</td></tr><tr><td colspan=\"2\">Individual Out-of-Pocket Max</td><td>$3,300</td></tr><tr><td colspan=\"2\">Family Out-of-Pocket Max</td><td>$6,600</td></tr><tr><td colspan=\"2\">Out-of-Pocket Max (Ind within Fam)</td><td>$3,300</td></tr><tr><td colspan=\"3\">Services</td></tr><tr><td>Office Visit</td><td></td><td>10%</td></tr><tr><td>Specialist Visit</td><td></td><td>10%</td></tr><tr><td>Preventive Care</td><td></td><td>$0 per visit</td></tr><tr><td>Emergency Room</td><td></td><td>10%</td></tr><tr><td>Urgent Care</td><td></td><td>10%</td></tr><tr><td>Diagnostic Lab/ X-Ray</td><td></td><td>10%</td></tr><tr><td>Hospital Inpatient</td><td></td><td>10%</td></tr><tr><td>Hospital Outpatient</td><td></td><td>10%</td></tr><tr><td colspan=\"3\">Rx</td></tr><tr><td>Retail Tier 1</td><td></td><td>$10 per script</td></tr><tr><td>Retail Tier 2</td><td></td><td>$30 per script</td></tr><tr><td>Retail Tier 3</td><td></td><td>$30 per script</td></tr><tr><td>Retail Tier 4</td><td></td><td>20% up to $250 1</td></tr></table>\n\n1\nAfter Deductible is Met\n\nLUMITY\nALERA GROUP COMPANY\n\nThis Benefits Guide offers a high-level overview of your available benefits. The official Plan Documents are what govern your official rights and coverage under each plan. \n This Benefits Guide offers a high-level overview of your available benefits. The official Plan Documents are what govern your official rights and coverage under each plan.",
    "page": "6"
},
{
    "content": "Automattic Inc., Benefits Guide 2025 Effective Period: January 01, 2025 December 31, 2025 \n Benefits At-A-Glance\n\nAutomattic Inc., Benefits Guide 2025 Effective Period: January 01, 2025 December 31, 2025\n\n## Dental\n\nIn-Network\n\n<table><tr><th></th><th></th><th>Aetna Dental PPO (Base)</th><th>Aetna Dental PPO (Buy\nUp)</th></tr><tr><td rowspan=\"4\">Per Month\nPremiums</td><td>Employee</td><td>$0.00</td><td>$8.00</td></tr><tr><td>+ Spouse</td><td>$0.00</td><td>$16.00</td></tr><tr><td>+ Child(ren)</td><td>$0.00</td><td>$16.00</td></tr><tr><td>+ Family</td><td>$0.00</td><td>$28.00</td></tr><tr><td colspan=\"4\">Deductible &amp; Benefit Maximums</td></tr><tr><td colspan=\"2\">Annual Maximum (Per Person)</td><td>$3,000</td><td>$5,000</td></tr><tr><td>Individual Deductible</td><td></td><td>$0</td><td>$0</td></tr><tr><td>Family Deductible</td><td></td><td>$0</td><td>$0</td></tr><tr><td>Orthodontia Maximum</td><td></td><td>$2,000</td><td>$3,000</td></tr><tr><td colspan=\"3\">Services</td><td></td></tr><tr><td>Preventive Care</td><td></td><td>0%</td><td>0%</td></tr><tr><td>Basic Care</td><td></td><td>10% 1</td><td>0%</td></tr><tr><td>Major Care</td><td></td><td>40%</td><td>30%</td></tr></table>\n\n## 9 Vision\n\n1\nAfter Deductible is Met\n\nIn-Network\n\n<table><tr><th colspan=\"2\"></th><th>Guardian Vision PPO\n(Base)</th><th>Guardian Vision PPO (Buy\nUp)</th></tr><tr><td rowspan=\"4\">Per Month\nPremiums</td><td>Employee</td><td>$0.00</td><td>$10.00</td></tr><tr><td>+ Spouse</td><td>$0.00</td><td>$22.00</td></tr><tr><td>+ Child(ren)</td><td>$0.00</td><td>$14.00</td></tr><tr><td>+ Family</td><td>$0.00</td><td>$26.00</td></tr><tr><td colspan=\"4\">Services</td></tr><tr><td colspan=\"2\">Exam Copay</td><td>$10 per year</td><td>$10 per year</td></tr><tr><td colspan=\"2\">Eyeglass Lenses Single</td><td>$25 per year</td><td>$25 per year</td></tr><tr><td colspan=\"2\">Eyeglass Lenses Bifocal</td><td>$25 per year</td><td>$25 per year</td></tr><tr><td colspan=\"2\">Eyeglass Lenses Trifocal</td><td>$25 per year</td><td>$25 per year</td></tr><tr><td colspan=\"2\">Frames</td><td>Up to $200 Allowance per year</td><td>Up to $250 Allowance per year</td></tr><tr><td colspan=\"2\">Elective Contacts</td><td>Up to $200</td><td>Up to $250 Allowance per year</td></tr><tr><td colspan=\"2\">Medically Necessary Contacts</td><td>Covered 100%</td><td>Covered 100%</td></tr></table>\n\nLUMITY\nALERA GROUP COMPANY\n\nThis Benefits Guide offers a high-level overview of your available benefits. The official Plan Documents are what govern your official rights and coverage under each plan. \n This Benefits Guide offers a high-level overview of your available benefits. The official Plan Documents are what govern your official rights and coverage under each plan.",
    "page": "7"
},
{
    "content": "Automattic Inc., Benefits Guide 2025 Effective Period: January 01, 2025 December 31, 2025 \n Benefits At-A-Glance\n\nAutomattic Inc., Benefits Guide 2025 Effective Period: January 01, 2025 December 31, 2025\n\n\n\n## Life & Disability\n\n<table><tr><th></th><th>Guardian Basic Life &amp; AD&amp;D</th></tr><tr><td>Benefit</td><td>2x Annual Salary</td></tr><tr><td>Benefit Maximum</td><td>$1,000,000</td></tr><tr><td>Guaranteed Issue</td><td>$1,000,000</td></tr></table>\n\n<table><tr><th></th><th>Guardian Short Term Disability</th></tr><tr><td>Benefit Percentage</td><td>60%</td></tr><tr><td>Maximum Weekly\nBenefit</td><td>$2,000</td></tr><tr><td>Maximum Benefit\nDuration</td><td>12 weeks</td></tr><tr><td>Elimination Period</td><td>3 days</td></tr></table>\n\n<table><tr><th></th><th>Guardian Long Term Disability</th></tr><tr><td>Benefit Percentage</td><td>60%</td></tr><tr><td>Maximum Monthly\nBenefit</td><td>$10,000</td></tr><tr><td>Maximum Benefit\nDuration</td><td>ADEA1 w/ SSNRA</td></tr><tr><td>Elimination Period</td><td>90 days</td></tr></table>\n\n<table><tr><th></th><th>Guardian Voluntary Life &amp; AD&amp;D</th></tr><tr><td>Employee Benefit</td><td>Increments of $10,000 up to $1,000,000</td></tr><tr><td>Spouse Benefit</td><td>Increments of $5,000 up to $100,000</td></tr><tr><td>Child Benefit</td><td>$1,000 up to $10,000</td></tr><tr><td>Employee\nGuaranteed Issue</td><td>$250,000</td></tr><tr><td>Spouse Guaranteed\nIssue</td><td>$25,000</td></tr><tr><td>Child Guaranteed\nIssue</td><td>$10,000</td></tr></table>\n\n\n\n## Worksite Benefits\n\n## Ladder Voluntary Life\n\nCoverage previously elected under Ladder is portable by default.\n\nMost experts recommend coverage of a minimum of 10x your annual salary to set your loved ones up with enough to cover living expenses, college tuition, and mortgage payments. Your Group Life offering is a great starting point, but for those with dependents it may not be enough.\n\nLadder makes it easy to fill that coverage gap, with a guaranteed digital-first application for up to $3m in coverage, the flexibility to adjust coverage over time as your needs change, and portability by default (meaning your policy stays in effect even if you change jobs).\n\nTo find out how much coverage you need and apply for term life insurance visit ladderlife.com/lumity. The digital process takes about 10 min to apply. You'll get an instant decision and if approved, the option to start protecting your family today.\n\nLadder Insurance Services, LLC (CA license OK22568; AR license  3000140372) distributes term life insurance products issued by multiple insurers - for further details see ladderlife.com. All insurance products are governed by the terms set forth in the applicable insurance policy. Each insurer has financial responsibility for its own products.\n\n- Easy digital application, with no medical exams up to $3m (only health questions are asked)\n\n- Policies are portable by default, and stay with you even if you change jobs\n\n- Available year-round (not just during open enrollment)\n\nGet started: ladderlife.com/lumity\n\nAetna Accident Plan\n\nYou must complete the Aetna Voluntary Portability form found under the Plan Resrouces link and return it to us along with payment the first premium for the portability coverage not later than 30 calendar days after your coverage under the policy ends. Portability coverage will be effective on the day after benefits under the policy end. If you have any questions, call member services at 1-800-800-8121 (TTY:711), Monday through Friday, 8 AM to 6 PM.\n\nAccident insurance provides a one-time lump sum cash payment if you suffer an accident covered under your policy. Examples may include broken bones, severe burns, or other emergency treatments. The purpose of this coverage is to help with the cost of an expensive, unexpected accident.\n\nAetna Voluntary Accident Plan Claim Submisson:\n\nLUMITY\nALERA GROUP COMPANY\n\nThis Benefits Guide offers a high-level overview of your available benefits. The official Plan Documents are what govern your official rights and coverage under each plan. \n ",
    "page": "8"
},
{
    "content": "Automattic Inc., Benefits Guide 2025 Effective Period: January 01, 2025 December 31, 2025 \n Benefits At-A-Glance\n\nAutomattic Inc., Benefits Guide 2025 Effective Period: January 01, 2025 December 31, 2025\n\nClaim form: Aetna Accident Claim Form pdf\n\nOnline web link: https://www12.aetna.com/AVOnlineClaimForm/welcome.aspx?productType=HOS\n\n## Aetna Critical Illness Plan\n\nYou must complete the Aetna Voluntary Portability form found under the Plan Resrouces link and return it to us along with payment the first premium for the portability coverage not later than 30 calendar days after your coverage under the policy ends. Portability coverage will be effective on the day after benefits under the policy end. If you have any questions, call member services at 1-800-800-8121 (TTY:711). Monday through Friday, 8 AM to 6 PM.\n\nCritical illness insurance provides a one-time lump sum cash payment if you are diagnosed with a covered condition listed in the policy. Examples may include cancer, strokes, heart attacks, strokes, etc. The purpose of this coverage is to help with the cost of treating and recovering from the specific condition.\n\nAetna Voluntary Critical Illness Plan Claim Submisson:\n\nClaim form: Aetna Criticall Illness Claim Form pdf\n\nOnline web link: https://www12.aetna.com/AVOnlineClaimForm/welcome.asp?productTypesHOS\n\n## Aetna Hospital Indemnity\n\nYou must complete the Aetna Voluntary Portability form found under the Plan Resrouces link and return it to us along with payment the first premium for the portability coverage not later than 30 calendar days after your coverage under the policy ends. Portability coverage will be effective on the day after benefits under the policy end. If you have any questions, call member services at 1-800-800-8121 (TTY:711). Monday through Friday, 8 AM to 6 PM.\n\nHospital insurance (also known as hospital indemnity insurance) provides you with benefits if you are hospitalized Examples may include if you are admitted for inpatient surgery or admitted for critical care. The purpose of this coverage is to help with the cost of an expensive hospital visit or admission\n\nAetna Voluntary Hospital indemnity Plan Claim Submisson:\n\nClaim form: Aetna Hospital Indemnity Claim Form.pdf Online web link: https://ww12.aetna.com/AVOnilineClaimForm/welcome.aspx?producTypesHOS\n\nLUMITY\nALERA GROUP COMPANY\n\nThis Benefits Guide offers a high-level overview of your available benefits. The official Plan Documents are what govern your official rights and coverage under each plan. \n This Benefits Guide offers a high-level overview of your available benefits. The official Plan Documents are what govern your official rights and coverage under each plan.",
    "page": "9"
},
{
    "content": "Automattic Inc., Benefits Guide 2025 Effective Period: January 01, 2025 December 31, 2025 \n Benefits At-A-Glance\n\nAutomattic Inc., Benefits Guide 2025 Effective Period: January 01, 2025 December 31, 2025\n\n\n\n## Tax Advantaged Accounts\n\n<table><tr><th>Employer\nContribution per\nMonth</th><th colspan=\"2\">Health Equity HSA</th></tr><tr><td>Employee</td><td>$137.50</td><td></td></tr><tr><td>Spouse</td><td>$275.00</td><td>A Health Savings Account (HSA) is like a \"healthcare 401(k).\" It's a tax-advantaged account you can dip into anytime to pay for qualified</td></tr><tr><td>Child(ren)</td><td>$275.00</td><td>out-of-pocket medical, dental, and vision expenses. Or, let it build to complement your retirement savings. You own the account, and\nfunds carry over year after year.</td></tr><tr><td>Family</td><td>$275.00</td><td></td></tr><tr><td>Contribution\nLimit(s)</td><td></td><td>HSA's have an annual contribution limit. While you must be enrolled in an HSA-eligible plan to contribute to your HSA, you can always\n\nspend your already existing HSA funds on qualified medical expenses.</td></tr><tr><td>Max Yearly\nContribution\n(Individual)</td><td>$4,300.00</td><td>HSA investment Oppurtunities- Investment threshold is $1,000. For additional information please see enclosed details and/or contact\nHealth Equity at (866) 346-5800.</td></tr><tr><td>Max Yearly\nContribution\n(Family)</td><td>$8,550.00</td><td>Health Equity- HSA Investment Guide pdf</td></tr></table>\n\n<table><tr><th>Contribution\nLimit(s)</th><th colspan=\"2\">HealthEquity Dependent Care FSA</th></tr><tr><td>Max Yearly\nContribution Married\nFiling Separately</td><td>$2,500.00</td><td>A Dependent Care Flexible Spending Account (DCFSA) also known as a Dependent Care Assistance Plan (DCAP) is an employer-\n\nowned, tax-advantaged spending account that you can use to pay for qualified child and elder care expenses while you're working,</td></tr><tr><td>Max Yearly</td><td></td><td>looking for work, or attending school full-time.</td></tr><tr><td>Contribution Married\nFiling Jointly or\nSingle Parent</td><td>$5,000.00</td><td>You need to incur all eligible dependent care expenses prior to the end of the plan year. Requests for reimbursements need to be filed\nbefore the end of the claims filing deadline for your plan, typically 75 days after the end of the plan year. Check with your employer for the\nexact date.</td></tr></table>\n\n\"Runout period, 3/31/2025 *For claims incurred prior to 12/31/2024 No rollover available for DCFA.'\n\n<table><tr><th>Contribution\nLimit(s)</th><th colspan=\"2\">HealthEquity Healthcare FSA</th></tr><tr><td>Annual Max\nContributions</td><td>$3,300.00</td><td>A General-Purpose Healthcare Flexible Spending Account (Healthcare FSA) can be used to pay for qualified out-of-pocket costs for</td></tr><tr><td>End of Year Policy</td><td>Up to $660\nrollover</td><td>medical, dental, and vision care, as well as many more expenses. Because the money you contribute to an FSA isn't taxed, you can\nreduce your overall healthcare expenses.</td></tr><tr><td></td><td></td><td>You typically need to incur all expenses prior to the end of the plan year. Requests for reimbursements need to be filed before the end of\nthe claims filing deadline for your plan, typically 75 days after the end of the plan year. Check with your employer for the exact date.</td></tr><tr><td></td><td></td><td>Your employer may also offer you a grace period OR a carryover option. If your employer offers you a grace period, you have an\nadditional 2\u00bd months after the end of the plan year to incur expenses. If your employer offers you the carryover option, you may carry a\nlimited amount from your account to the next plan year.</td></tr><tr><td></td><td></td><td>\"Runout period, 3/31/2025. *For claims incurred prior to 12/31/2024 2024 rollover funds will not be available/posted into the member's\naccount until 5/31/2025.*</td></tr></table>\n\n<table><tr><th colspan=\"3\">Contribution\nLimit(s) HealthEquity Limited FSA</th></tr><tr><td>Annual Max\nContributions</td><td>$3,300.00</td><td rowspan=\"2\">A Limited Prupose flexible spending account can be used to pay for qualified out-of-pocket costs for dental, and vision care. Because the\n\nmoney you contribute to an FSA isn't taxed, you can reduce your overall healthcare expenses.</td></tr><tr><td>End of Year Policy</td><td>Up to $660\nrollover</td></tr></table>\n\nLUMITY\nGROUP COMPANY\n\nThis Benefits Guide offers a high-level overview of your available benefits. The official Plan Documents are what govern your official rights and coverage under each plan. \n This Benefits Guide offers a high-level overview of your available benefits. The official Plan Documents are what govern your official rights and coverage under each plan.",
    "page": "10"
},
{
    "content": "Automattic Inc., Benefits Guide 2025 Effective Period: January 01, 2025 December 31, 2025 \n Benefits At-A-Glance\n\nAutomattic Inc., Benefits Guide 2025 Effective Period: January 01, 2025 December 31, 2025\n\nYou typically need to incur all expenses prior to the end of the plan year. Requests for reimbursements need to be filed before the end of the claims filing deadline for your plan, typically 75 days after the end of the plan year. Check with your employer for the exact date.\n\nYour employer may also offer you a grace period OR a carryover option. If your employer offers you a grace period, you have an additional 2\u00bd months after the end of the plan year to incur expenses. If your employer offers you the carryover option, you may carry a limited amount from your account to the next plan year.\n\n*75 day runout period, 3/31/2025. *For claims incurred prior to 12/31/2024. 2024 rollover funds will not be available/posted into the member's account until 4/30/2025.*\n\n\n\n401(k)\n\n<table><tr><th></th><th>Betterment</th><th>401K</th><th>Plan</th></tr><tr><td>Annual Max\nContribution</td><td>$23,500.00</td><td></td><td>A 401(k) is an employer sponsored, tax-advantaged account that allows you to put aside money for retirement. A set percentage of your</td></tr><tr><td>Annual Catchup\nContribution 50 or\nOlder</td><td>$7,500.00</td><td></td><td>choosing is automatically taken out of each paycheck and invested in a 401(k) account. Your contributions will be invested in stocks,\n\nbonds, mutual funds, etc. that you can pick yourself or that the vendor will choose on your behalf.</td></tr></table>\n\nLUMITY\nALERA GROUP COMPANY\n\nThis Benefits Guide offers a high-level overview of your available benefits. The official Plan Documents are what govern your official rights and coverage under each plan. \n This Benefits Guide offers a high-level overview of your available benefits. The official Plan Documents are what govern your official rights and coverage under each plan.",
    "page": "11"
},
{
    "content": "Automattic Inc., Benefits Guide 2025 Effective Period: January 01, 2025 December 31, 2025 \n Important Benefit Contact Resources\n\nAutomattic Inc., Benefits Guide 2025 Effective Period: January 01, 2025 December 31, 2025\n\n<table><tr><th>Contact</th><th>Phone</th><th>Website/Email</th></tr><tr><td>InsightTimer</td><td></td><td>https://insighttimer.com/</td></tr><tr><td>Guardian</td><td></td><td>https://www.guardianlife.com/life-insurance</td></tr><tr><td>Kaiser Permanente HI</td><td>1-800-966-5955</td><td>https://healthy.kaiserpermanente.org/hawaii</td></tr><tr><td>Ten Percent Happier</td><td></td><td>https://www.tenpercent.com/</td></tr><tr><td>Automattic</td><td>1-844-258-6489</td><td>https://automattic.com/</td></tr><tr><td>GoodRx</td><td>1-855-268-2822</td><td>https://www.goodrx.com/</td></tr><tr><td>Kaiser Permanente Southern CA</td><td>1-800-464-4000</td><td>https://healthy.kaiserpermanente.org/southern-california</td></tr><tr><td>Aetna</td><td></td><td>https://www.aetna.com</td></tr><tr><td>Automattic</td><td>1-844-258-6489</td><td>https://automattic.com/</td></tr><tr><td>Kaiser Permanente Rx Delivery/Coupons</td><td>1-800-464-4000</td><td>https://healthy.kaiserpermanente.org/learn/pharmacy</td></tr><tr><td>Betterment</td><td></td><td>https://www.betterment.com/retirement, https:/www.betterment.com</td></tr><tr><td>Kaiser Permanente Northern CA</td><td>1-800-464-4000</td><td>https://healthy.kaiserpermanente.org/northern-california</td></tr><tr><td>Headspace</td><td></td><td>https://www.headspace.com/</td></tr><tr><td>Aetna</td><td>1-844-365-7373</td><td>https://www.aetna.com</td></tr><tr><td>Health Equity</td><td></td><td>https://healthequity.com</td></tr><tr><td>Kaiser Permanente Telehealth</td><td></td><td>https://healthy.kaiserpermanente.org/learn/how-to-use-telehealth</td></tr><tr><td>Automattic</td><td>1-844-258-6489</td><td>https://automattic.com/</td></tr><tr><td>Pet Benefit Solutions</td><td></td><td>https://www.petbenefits.com/</td></tr><tr><td>Aetna</td><td>1-800-872-3862</td><td>https://www.aetna.com</td></tr><tr><td>HMAA</td><td>1-808-941-4622</td><td>https://www.hmaa.com/</td></tr><tr><td>Aetna</td><td></td><td>https://www.aetna.com/services/telehealth.html</td></tr><tr><td>Ladder</td><td></td><td>https://www.ladderlife.com/</td></tr><tr><td>Aetna</td><td></td><td>https://www.aetna.com/individuals-families/pharmacy/rx-home-delivery.html</td></tr><tr><td>Legal Shield</td><td></td><td>https://www.legalshield.com/</td></tr><tr><td>Guardian (VSP)</td><td>1-800-627-4200</td><td>https://www.guardianlife.com/</td></tr><tr><td>Aetna</td><td>1-877-204-9186</td><td>https://www.aetna.com</td></tr><tr><td>Aetna</td><td></td><td>https://www.aetna.com</td></tr><tr><td>Waking Up</td><td></td><td>https://www.wakingup.com/</td></tr><tr><td>Kaiser Permanente Colorado</td><td></td><td>https://healthy.kaiserpermanente.org/</td></tr><tr><td>Calm</td><td></td><td>https://www.calm.com</td></tr></table>\n\nLUMITY\nALERA GROUP COMPANY\n\nThis Benefits Guide offers a high-level overview of your available benefits. The official Plan Documents are what govern your official rights and coverage under each plan. \n This Benefits Guide offers a high-level overview of your available benefits. The official Plan Documents are what govern your official rights and coverage under each plan.",
    "page": "12"
}
] 
"""

STAGE_2_PROMPT = """
AI Employee Assistant Prompt v3.0 - Stage 2
    
description: AI assistant built into PlanYear, focused on U.S. employee benefits.
capabilities:
- Answers employee benefits questions using structured context
- Outputs markdown-formatted answers with cited sources

context_sources:
- QUESTION: Current user query
- CHAT_HISTORY: Last 4 chat messages
- EMPLOYER_BENEFITS_KNOWLEDGE_BASE_DOCUMENTS:
    description: >
        Documents broken into page-level objects with metadata.

allowed:
    - Employee benefits questions using provided context
    - General employee benefits questions from training
not_allowed:
    - Personal, legal, or financial advice
    - Questions outside employee benefits

classification_task:
labels:
    - Not allowed
    - Follow-up
    - New question
    - Unclear
output_format: "CLASSIFICATION: <label>"

output_format:
tone: Friendly and professional
style: Markdown
structure:
    - Rewrite technical DBG into plain English
    - Append "Contact x@y.com if you need additional information."
    - Do not embed links inside main content
    - Only cite sources that were actually used

example_response:
format: Markdown
content: 
    HSA Contribution Limits for 2025

    - **Individual Coverage:** $4,300  
    - **Family Coverage:** $8,550  
    - **Catch-Up (55+):** $1,000 extra

    You must be enrolled in a qualified HDHP:
    - Min Deductible: $1,650 (individual), $3,300 (family)  
    - Max Out-of-Pocket: $8,300 (individual), $16,600 (family)

    **Sources:**  
    
    • 2025_HSA_Guide.pdf, pages 10 and 12
    • Carrier_Summary_Sheet.pdf, page 3
    
    Contact x@y.com if you need additional information.
    
**IMPORTANT RULES:**
1. Only use the EMPLOYER_BENEFITS_KNOWLEDGE_BASE_DOCUMENTS provided below to answer questions
2. If the DBG doesn't contain sufficient information to answer the question, respond with exactly: "I'm sorry, I don't have enough information to answer that. Please contact x@y.com."
3. If you can answer from the knowledge base, provide a clear, helpful response and cite page numbers
4. Do not answer questions outside of employee benefits
5. You have to follow the example_response in terms of formatting
6. Sources should be cited at the very bottom of your response with appropriate break lines between each source
"""

# ---- Models ----
class Query(BaseModel):
    question: str
    user_id: UUID

# ---- Helpers ----
def get_chat_history(user_id: str):
    res = supabase.table("chat_history") \
        .select("message") \
        .filter("user_id", "eq", user_id) \
        .order("created_at", desc=True) \
        .limit(4) \
        .execute()

    rows = res.data or []
    # Just join the last 4 messages without role labeling
    return "\n".join(r["message"] for r in reversed(rows))

def get_vector_knowledge(question: str):
    print("\n[DEBUG] Searching Vectorize for question:", question)

    res = requests.post(
        VECTORIZE_URL,
        headers={
            "Authorization": f"Bearer {VECTORIZE_KEY}",
            "Content-Type": "application/json"
        },
        json={"question": question, "numResults": 10}
    )

    try:
        data = res.json()
    except Exception as e:
        print("[ERROR] Could not parse Vectorize JSON:", e)
        print("[RAW RESPONSE]", res.text)
        return ""

    # Debugging output
    print("\n[DEBUG] Full Vectorize response:")
    import json as _json
    print(_json.dumps(data, indent=2))

    matches = data.get("matches", [])
    if not matches:
        print("[DEBUG] No matches found for query.")
        return ""

    print("\n[DEBUG] Top Matches (score + short content preview):")
    for m in matches:
        score = m.get("score")
        content = m.get("payload", {}).get("content", "")
        preview = content[:120].replace("\n", " ") + "..." if content else ""
        print(f"  - Score: {score:.4f} | Preview: {preview}")

    return "\n\n".join(m["payload"]["content"] for m in matches if "payload" in m)

# ---- Main workflow ----
def run_workflow(question, user_id):
    chat_history = get_chat_history(user_id)

    # Stage 1 prompt with question + chat history appended
    stage1_prompt = f"""{STAGE_1_PROMPT}

**CURRENT QUESTION:**
{question}

**PAST CHAT HISTORY**
{chat_history or "None"}
"""

    first_agent = Agent(
        role="Stage 1 Agent",
        goal="Answer benefits questions",
        backstory="An expert in employee benefits who helps answer employee questions using PlanYear's knowledge base.",
        allow_delegation=False,
        verbose=True, 
        llm=gemini_llm
    )
    first_task = Task(
        description=stage1_prompt,
        agent=first_agent,
        expected_output="A well-formatted markdown answer to the user's benefits question with sources cited at the bottom."
    )
    first_crew = Crew(agents=[first_agent], tasks=[first_task], verbose=True)
    first_response = first_crew.kickoff()

    # CrewOutput has `.raw` and `.outputs` attributes
    if hasattr(first_response, "raw"):
        output_text = str(first_response.raw)
    else:
        output_text = str(first_response)

    if "insufficient information" in output_text.lower():
        vector_knowledge = get_vector_knowledge(question)
        stage2_prompt = f"""{STAGE_2_PROMPT}

**CURRENT QUESTION:**
{question}

**CHAT HISTORY:**
{chat_history or "None"}

**KNOWLEDGE BASE DOCUMENTS:**
{vector_knowledge}
"""
        second_agent = Agent(
            role="Stage 2 Agent",
            goal="Answer benefits questions",
            backstory="A benefits research specialist who searches the knowledge base to find detailed answers.",
            allow_delegation=False,
            verbose=True, 
            llm=gemini_llm
        )
        second_task = Task(
            description=stage2_prompt,
            agent=second_agent,
            expected_output="A well-formatted markdown answer to the user's benefits question with sources cited at the bottom."
        )
        second_crew = Crew(agents=[second_agent], tasks=[second_task], verbose=True)
        return second_crew.kickoff()

    return first_response

# ---- API endpoint ----
@app.post("/ask")
def ask(query: Query):
    answer = run_workflow(query.question, query.user_id)
    return {"answer": answer}
