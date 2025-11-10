from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Union
from team import team_data
from standing_east import standings_data_eastern
from standing_west import standings_data_western
from game import games_data
import os
from dotenv import load_dotenv
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

# Load environment variables
load_dotenv()


app = FastAPI(title="NBA API", version="1.0.0")

# CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Pydantic Models
class Player(BaseModel):
    name: str
    position: str
    height: str
    weight: str
    last_attended: str
    country: str

class Team(BaseModel):
    id: int
    name: str
    roster: List[Player]

class Standing(BaseModel):
    Team: str
    W: int
    L: int
    PCT: float
    GB: str
    L10: str
    Strk: str

# API Endpoints

@app.get("/")
def read_root():
    """Welcome endpoint"""
    return {
        "message": "Welcome to NBA API",
        "endpoints": {
            "/teams": "Get all teams",
            "/teams/{team_id}": "Get team by ID",
            "/teams/name/{team_name}": "Get team by name",
            "/standings/eastern": "Get Eastern Conference standings"
        }
    }

@app.get("/teams", response_model=List[Dict])
def get_all_teams():
    """Get all NBA teams with basic info (id and name only)"""
    return [{"id": team["id"], "name": team["name"]} for team in team_data]

@app.get("/teams/{team_id}", response_model=Team)
def get_team_by_id(team_id: int):
    """Get team details by ID"""
    team = next((t for t in team_data if t["id"] == team_id), None)
    if team is None:
        raise HTTPException(status_code=404, detail=f"Team with ID {team_id} not found")
    return team

@app.get("/teams/name/{team_name}", response_model=Team)
def get_team_by_name(team_name: str):
    """Get team details by name (case-insensitive)"""
    team = next((t for t in team_data if t["name"].lower() == team_name.lower()), None)
    if team is None:
        raise HTTPException(status_code=404, detail=f"Team '{team_name}' not found")
    return team

@app.get("/teams/{team_id}/roster", response_model=List[Player])
def get_team_roster(team_id: int):
    """Get roster for a specific team"""
    team = next((t for t in team_data if t["id"] == team_id), None)
    if team is None:
        raise HTTPException(status_code=404, detail=f"Team with ID {team_id} not found")
    return team["roster"]

@app.get("/standings/eastern", response_model=List[Standing])
def get_eastern_standings():
    """Get Eastern Conference standings"""
    return standings_data_eastern

@app.get("/standings/western", response_model=List[Standing])
def get_eastern_standings():
    """Get Western Conference standings"""
    return standings_data_western

@app.get("/teams/{team_id}/standing")
def get_team_standing(team_id: int):
    """Get standing info for a specific team"""
    team = next((t for t in team_data if t["id"] == team_id), None)
    if team is None:
        raise HTTPException(status_code=404, detail=f"Team with ID {team_id} not found")

    # Find team in standings
    standing = next((s for s in standings_data_eastern if s["Team"] == team["name"]), None)
    if standing is None:
        raise HTTPException(status_code=404, detail=f"Standing data not found for {team['name']}")

    return {
        "team": team["name"],
        "standing": standing
    }

@app.get("/search/players")
def search_players(name: Optional[str] = None, position: Optional[str] = None, country: Optional[str] = None):
    """Search players by name, position, or country"""
    results = []

    for team in team_data:
        for player in team["roster"]:
            match = True
            if name and name.lower() not in player["name"].lower():
                match = False
            if position and position.upper() not in player["position"].upper():
                match = False
            if country and country.lower() not in player["country"].lower():
                match = False

            if match:
                results.append({
                    "player": player,
                    "team": team["name"]
                })

    if not results:
        raise HTTPException(status_code=404, detail="No players found matching the criteria")

    return {
        "count": len(results),
        "results": results
    }

class Game(BaseModel):
    team1: str
    team2: str
    date: str
    location: str

@app.get("/games", response_model=List[Game])
def get_all_games():
    """Get all scheduled games"""
    return games_data

@app.get("/games/search", response_model=List[Game])
def search_games(team: Optional[str] = None, date: Optional[str] = None):
    """Search games by team name or date"""
    results = []
    for game in games_data:
        match = True
        if team and team.lower() not in (game["team1"].lower() + game["team2"].lower()):
            match = False
        if date and date != game["date"]:
            match = False
        if match:
            results.append(game)
    if not results:
        raise HTTPException(status_code=404, detail="No games found matching the criteria")
    return results

# Chatbot Models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

class ChatResponse(BaseModel):
    response: str

# Initialize Watsonx AI client
def get_watsonx_client():
    credentials = Credentials(
        url=os.getenv("WATSONX_URL"),
        api_key=os.getenv("WATSONX_API_KEY")
    )
    return APIClient(credentials)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with NBA assistant using IBM Watsonx AI"""
    try:
        client = get_watsonx_client()

        params = {
            "max_tokens": 500,
            "temperature": 0.7
        }

        model = ModelInference(
            model_id="ibm/granite-3-8b-instruct",
            api_client=client,
            params=params,
            project_id=os.getenv("WATSONX_PROJECT_ID"),
            space_id=None,
            verify=False,
        )

        # Convert request messages to Watsonx format
        messages = []
        for msg in request.messages:
            if msg.role == "system":
                messages.append({
                    "role": "system",
                    "content": msg.content
                })
            elif msg.role == "user":
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": msg.content}]
                })
            elif msg.role == "assistant":
                messages.append({
                    "role": "assistant",
                    "content": msg.content
                })

        # Get response from Watsonx
        response = model.chat(messages=messages)

        # Extract text from response
        response_text = ""
        if hasattr(response, 'choices') and len(response.choices) > 0:
            response_text = response.choices[0].message.content
        elif isinstance(response, dict) and 'choices' in response:
            response_text = response['choices'][0]['message']['content']
        else:
            response_text = str(response)

        return ChatResponse(response=response_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


if __name__ == "__main__":
    import sys
    import asyncio
    import uvicorn

    # Windows-specific fix for WinError 10014
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    uvicorn.run(app, host="127.0.0.1", port=8000)
