from fastapi import FastAPI
from pydantic import BaseModel
from crewai import Agent, Crew, Task

app = FastAPI(
    title="CrewAI Backend",
    description="API for analyzing user preferences and recommending plans.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

@app.get("/")
async def root():
    return {"message": "CrewAI backend is running!"}

class PreferenceRequest(BaseModel):
    preferences: str
    plans: list

def format_plans(plans_list):
    formatted = ""
    for plan in plans_list:
        formatted += (
            f"\n- Name: {plan.get('name')}\n"
            f"  Deductible: {plan.get('deductible')}\n"
            f"  Monthly Premium: {plan.get('monthly_premium')}\n"
            f"  Network: {plan.get('network')}\n"
        )
    return formatted

@app.post("/run")
async def run_crew(req: PreferenceRequest):
    user_text = req.preferences
    plans = req.plans
    formatted_plans = format_plans(plans)
    print(formatted_plans)

    # Define agents
    preference_analyzer = Agent(
        name="Preference Analyzer",
        role="Summarize employee's key healthcare plan preferences",
        goal="Extract and clearly list the most important preferences from the user input",
        backstory="You excel at understanding employee needs and summarizing them clearly.",
    )

    plan_selector = Agent(
        name="Plan Selector",
        role="Match and rank plans",
        goal="Use employee preferences and available plan data to select the best option, providing detailed reasoning and a ranked list",
        backstory="You carefully evaluate trade-offs and match plans to the user's exact needs.",
    )

    final_recommender = Agent(
        name="Final Recommender",
        role="Explain recommendation clearly",
        goal="Paraphrase the selected plan recommendation into friendly, empathetic language for the employee",
        backstory="You are excellent at simplifying complex information and making the employee feel confident in their choice.",
    )

    task1 = Task(
        description=f"""Analyze the following user preferences and produce a structured list of key criteria...
User Preferences:
{user_text}
""",
        expected_output="A clear, numbered list summarizing the key criteria.",
        agent=preference_analyzer,
    )

    task2 = Task(
        description=f"""Use the following plans data: {formatted_plans}
and the extracted employee preferences from the Preference Analyzer.
Rank the plans from best to worst based on how they match the employee's preferences...
""",
        expected_output="A ranked list with reasoning and a final plan selection.",
        agent=plan_selector,
    )

    task3 = Task(
        description="""Take the ranked plans and final selection from the Plan Selector.
Rephrase it into friendly, empathetic language that can be shown to the employee on the final answer page...
""",
        expected_output="A final friendly recommendation paragraph.",
        agent=final_recommender,
    )

    crew = Crew(
        agents=[preference_analyzer, plan_selector, final_recommender],
        tasks=[task1, task2, task3],
        verbose=True,
    )

    final_recommendation = crew.kickoff()

    return {"result": final_recommendation}
