#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SlotGenius AI Platform - Backend API
Complete FastAPI implementation ready for Railway deployment
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import os
import asyncio
import asyncpg
import redis.asyncio as redis
import hashlib
import secrets
import jwt
import json
import httpx
from enum import Enum
import logging
from contextlib import asynccontextmanager

# =====================================================
# CONFIGURATION
# =====================================================
# Environment variables for Railway
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/slotgenius")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(32))
ALGORITHM = "HS256"

# Meta Pixel Configuration
META_PIXEL_ID = "586986157788412"
META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN", "EAAI6cNS7kTkBOwFEO0Bf1YO6796fbkLTYEQZBG1c7RkMBoLzCu8hkqYxNwDLmZAbsCnlakZBJsmkfkCq4TerjNU8oWhYjtmxKfjnLDjpgi4if8mNLZCF06qFfZCpJlggiGztMgsC48Ve6YgxhIcMb1azDICbjavtr1G31bPZCZBrVxZA1TADjMZBjVNNcZCNqlnuVjNAZDZD")
META_API_VERSION = "v18.0"

# Affise Configuration
AFFISE_BASE_URL = "https://serversys.media-412.com/click"
AFFISE_PID = "253"

# Casino Commissions (Updated)
CASINO_COMMISSIONS = {
    "Nine Casinò": 80.0,
    "Green Luck": 100.0,
    "Bassbet": 90.0,
    "PriBet": 80.0
}

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================
# DATABASE CONNECTION
# =====================================================
class Database:
    pool: asyncpg.Pool = None
    
    @classmethod
    async def connect(cls):
        cls.pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=5,
            max_size=20,
            command_timeout=60
        )
        
        # Create tables if not exist
        async with cls.pool.acquire() as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    email VARCHAR(255) UNIQUE NOT NULL,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    telegram_id BIGINT,
                    subscription_tier VARCHAR(20) DEFAULT 'free',
                    scans_remaining INTEGER DEFAULT 6,
                    ultra_rare_eligible BOOLEAN DEFAULT FALSE,
                    ltv_score REAL DEFAULT 0,
                    total_scans INTEGER DEFAULT 0,
                    total_conversions INTEGER DEFAULT 0,
                    total_revenue REAL DEFAULT 0,
                    utm_source VARCHAR(50),
                    utm_medium VARCHAR(50),
                    utm_campaign VARCHAR(100),
                    utm_content VARCHAR(100),
                    utm_term VARCHAR(100),
                    fbclid VARCHAR(255),
                    affise_clickid VARCHAR(255),
                    created_at TIMESTAMP DEFAULT NOW(),
                    last_active_at TIMESTAMP DEFAULT NOW()
                )
            ''')
            
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS anomalies (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID REFERENCES users(id),
                    anomaly_code VARCHAR(20) UNIQUE NOT NULL,
                    type VARCHAR(20) NOT NULL,
                    casino_name VARCHAR(50) NOT NULL,
                    casino_offer_id VARCHAR(20) NOT NULL,
                    slot_name VARCHAR(100) NOT NULL,
                    rtp_value REAL NOT NULL,
                    confidence INTEGER NOT NULL,
                    volatility VARCHAR(20),
                    expected_payout_range VARCHAR(50),
                    optimal_bet REAL,
                    region VARCHAR(50),
                    expires_at TIMESTAMP NOT NULL,
                    claimed_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            ''')
            
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS casino_clicks (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID REFERENCES users(id),
                    anomaly_id UUID REFERENCES anomalies(id),
                    casino_name VARCHAR(50),
                    casino_offer_id VARCHAR(20),
                    affise_url TEXT,
                    clicked_at TIMESTAMP DEFAULT NOW()
                )
            ''')
            
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS conversions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID REFERENCES users(id),
                    conversion_type VARCHAR(20),
                    casino_name VARCHAR(50),
                    offer_id VARCHAR(20),
                    conversion_value REAL,
                    user_deposit_amount REAL,
                    postback_data JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            ''')
            
            # Create indexes
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_anomalies_user_expires ON anomalies(user_id, expires_at)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_conversions_user ON conversions(user_id)')
    
    @classmethod
    async def disconnect(cls):
        await cls.pool.close()

# Redis Connection
redis_client: redis.Redis = None

# =====================================================
# MODELS
# =====================================================
class UserRole(str, Enum):
    FREE = "free"
    PREMIUM = "premium"
    VIP = "vip"

class AnomalyType(str, Enum):
    STANDARD = "standard"
    ULTRA_RARE = "ultra_rare"

class UserRegister(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=20)
    password: str = Field(..., min_length=8)
    utm_source: Optional[str] = None
    utm_medium: Optional[str] = None
    utm_campaign: Optional[str] = None
    utm_content: Optional[str] = None
    utm_term: Optional[str] = None
    fbclid: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class AnomalyScanRequest(BaseModel):
    regions: List[str]
    scan_type: str = "all"  # all, standard, ultra_rare

# =====================================================
# SECURITY
# =====================================================
security = HTTPBearer()

async def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(hours=24))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# =====================================================
# META PIXEL SERVICE
# =====================================================
class MetaPixelService:
    @staticmethod
    async def track_event(user_id: str, event_name: str, custom_data: Dict = None):
        """Send server-side events to Meta Pixel"""
        try:
            if not META_PIXEL_ID or not META_ACCESS_TOKEN:
                logger.warning(f"Meta Pixel not configured for event {event_name}")
                return False
                
            # Generate event_id for deduplication
            event_id = hashlib.sha256(f"{user_id}_{event_name}_{int(datetime.utcnow().timestamp())}".encode()).hexdigest()
            
            # Prepare event data
            event_data = {
                "data": [{
                    "event_name": event_name,
                    "event_time": int(datetime.utcnow().timestamp()),
                    "event_id": event_id,
                    "action_source": "website",
                    "user_data": {
                        "external_id": hashlib.sha256(str(user_id).encode()).hexdigest(),
                        "client_user_agent": "SlotGenius AI Platform"
                    },
                    "custom_data": custom_data or {}
                }]
            }
            
            # IMPORTANT: Only Purchase events should have value
            if event_name != "Purchase" and "value" in event_data["data"][0]["custom_data"]:
                del event_data["data"][0]["custom_data"]["value"]
            
            # Add fbclid if available
            async with Database.pool.acquire() as conn:
                user = await conn.fetchrow('SELECT fbclid FROM users WHERE id = $1', user_id)
                if user and user['fbclid']:
                    event_data["data"][0]["user_data"]["fbc"] = f"fb.1.{int(datetime.utcnow().timestamp())}_{user['fbclid']}"
            
            # Send to Meta
            url = f"https://graph.facebook.com/{META_API_VERSION}/{META_PIXEL_ID}/events"
            params = {"access_token": META_ACCESS_TOKEN}
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, params=params, json=event_data)
                
                if response.status_code == 200:
                    logger.info(f"✅ Meta event '{event_name}' sent for user {user_id}")
                    return True
                else:
                    logger.error(f"❌ Meta event error: {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Meta event error: {e}")
            return False

# =====================================================
# CASINO ROUTING SERVICE
# =====================================================
class CasinoRouter:
    CASINO_OFFERS = {
        "Bassbet": {"offer_id": "184", "commission": 90.0, "url": "bassbet"},
        "Green Luck": {"offer_id": "179", "commission": 100.0, "url": "greenluck"},
        "Nine Casinò": {"offer_id": "155", "commission": 80.0, "url": "ninecasino"},
        "PriBet": {"offer_id": "151", "commission": 80.0, "url": "pribet"}
    }
    
    REGION_CASINOS = {
        "lombardia": ["Bassbet", "Green Luck", "Nine Casinò", "PriBet"],
        "lazio": ["Green Luck", "Bassbet", "PriBet", "Nine Casinò"],
        "veneto": ["Nine Casinò", "Bassbet", "Green Luck", "PriBet"],
        "piemonte": ["Green Luck", "PriBet", "Bassbet", "Nine Casinò"],
        "campania": ["Bassbet", "Green Luck", "Nine Casinò", "PriBet"],
        "emilia": ["Bassbet", "Green Luck", "PriBet", "Nine Casinò"],
        "toscana": ["Nine Casinò", "Green Luck", "Bassbet", "PriBet"],
        "sicilia": ["Bassbet", "Nine Casinò", "Green Luck", "PriBet"]
    }
    
    @classmethod
    async def get_optimal_casino(cls, region: str, user_id: str) -> Dict:
        """Select best casino based on region and user history"""
        region_key = region.lower().replace('-', '').replace(' ', '').replace("'", "")
        casinos = cls.REGION_CASINOS.get(region_key, list(cls.CASINO_OFFERS.keys()))
        
        # Get user conversion history
        async with Database.pool.acquire() as conn:
            history = await conn.fetch("""
                SELECT casino_name, COUNT(*) as conversions
                FROM conversions
                WHERE user_id = $1
                GROUP BY casino_name
            """, user_id)
        
        # Avoid casinos where user already converted
        converted_casinos = [h['casino_name'] for h in history]
        available_casinos = [c for c in casinos if c not in converted_casinos]
        
        if not available_casinos:
            available_casinos = casinos  # Reset if all used
        
        # Select first available casino for the region
        selected_casino = available_casinos[0]
        casino_data = cls.CASINO_OFFERS[selected_casino]
        
        return {
            'name': selected_casino,
            'offer_id': casino_data['offer_id'],
            'commission': casino_data['commission']
        }
    
    @classmethod
    async def generate_affise_url(cls, user_id: str, casino_name: str, anomaly_id: str) -> str:
        """Generate Affise tracking URL"""
        casino_data = cls.CASINO_OFFERS.get(casino_name, {"offer_id": "100"})
        
        # Get user tracking data
        async with Database.pool.acquire() as conn:
            user = await conn.fetchrow("""
                SELECT affise_clickid, utm_source, utm_campaign, fbclid
                FROM users WHERE id = $1
            """, user_id)
        
        # Generate unique click ID if not exists
        affise_clickid = user['affise_clickid'] if user and user['affise_clickid'] else f"sg_{user_id}_{int(datetime.utcnow().timestamp())}"
        
        params = {
            'pid': AFFISE_PID,
            'offer_id': casino_data['offer_id'],
            'sub1': affise_clickid,
            'sub2': user['utm_source'] if user and user['utm_source'] else 'platform',
            'sub3': user['utm_campaign'] if user and user['utm_campaign'] else 'web',
            'sub4': user['fbclid'] if user and user['fbclid'] else '',
            'sub5': str(user_id),  # IMPORTANT: User ID for postback
            'sub6': anomaly_id
        }
        
        url_params = '&'.join([f"{k}={v}" for k, v in params.items() if v])
        return f"{AFFISE_BASE_URL}?{url_params}"

# =====================================================
# ANOMALY ENGINE
# =====================================================
class AnomalyEngine:
    SLOTS = {
        "Bassbet": [
            {"name": "Book of Ra Deluxe", "exp_win": "€180 - €320", "volatility": "Alta"},
            {"name": "Gates of Olympus", "exp_win": "€220 - €410", "volatility": "Alta"},
            {"name": "Sweet Bonanza", "exp_win": "€195 - €365", "volatility": "Media"}
        ],
        "Green Luck": [
            {"name": "Starburst", "exp_win": "€155 - €245", "volatility": "Bassa"},
            {"name": "Gonzo's Quest", "exp_win": "€185 - €315", "volatility": "Media"},
            {"name": "Reactoonz", "exp_win": "€205 - €385", "volatility": "Alta"}
        ],
        "Nine Casinò": [
            {"name": "Zeus of Olympus", "exp_win": "€215 - €395", "volatility": "Alta"},
            {"name": "Sugar Buzz", "exp_win": "€185 - €325", "volatility": "Media"},
            {"name": "Fruity X125", "exp_win": "€235 - €425", "volatility": "Alta"}
        ],
        "PriBet": [
            {"name": "Dog House", "exp_win": "€210 - €375", "volatility": "Media"},
            {"name": "Wolf Gold", "exp_win": "€185 - €315", "volatility": "Media"},
            {"name": "Great Rhino", "exp_win": "€245 - €405", "volatility": "Alta"}
        ]
    }
    
    ULTRA_RARE_SLOTS = {
        "Bassbet": {"name": "Book of Ra Deluxe QUANTUM", "exp_win": "€285 - €465"},
        "Green Luck": {"name": "Gonzo's Quest NEURAL", "exp_win": "€315 - €545"},
        "Nine Casinò": {"name": "Zeus ULTRA POWER", "exp_win": "€395 - €675"},
        "PriBet": {"name": "Great Rhino DIAMOND", "exp_win": "€355 - €595"}
    }
    
    @classmethod
    async def create_anomaly(cls, user_id: str, region: str, scan_type: str) -> Optional[Dict]:
        """Create a new anomaly"""
        import random
        
        # Check user eligibility
        async with Database.pool.acquire() as conn:
            user = await conn.fetchrow("""
                SELECT scans_remaining, ultra_rare_eligible, ltv_score
                FROM users WHERE id = $1
            """, user_id)
            
            if not user or user['scans_remaining'] <= 0:
                return None
        
        # Determine anomaly type
        if scan_type == 'ultra_rare' and not user['ultra_rare_eligible']:
            return None
        
        # Decide if anomaly is found (not guaranteed)
        if scan_type == 'ultra_rare':
            found_chance = 0.7  # 70% chance for ultra rare when eligible
        elif scan_type == 'standard':
            found_chance = 0.8  # 80% chance for standard
        else:  # all
            found_chance = 0.85  # 85% chance for any type
        
        if random.random() > found_chance:
            return None  # No anomaly found
        
        # Determine actual type
        if scan_type == 'ultra_rare':
            is_ultra_rare = True
        elif scan_type == 'standard':
            is_ultra_rare = False
        else:  # all
            is_ultra_rare = user['ultra_rare_eligible'] and random.random() < 0.2
        
        # Get casino for region
        casino_info = await cls.get_optimal_casino(region, user_id)
        casino_name = casino_info['name']
        
        # Select slot
        if is_ultra_rare:
            slot_data = cls.ULTRA_RARE_SLOTS.get(casino_name)
            slot = {
                'name': slot_data['name'],
                'exp_win': slot_data['exp_win'],
                'volatility': 'Molto Alta'
            }
        else:
            slots = cls.SLOTS.get(casino_name, cls.SLOTS['Bassbet'])
            slot = random.choice(slots)
        
        # Generate anomaly data
        anomaly_type = AnomalyType.ULTRA_RARE if is_ultra_rare else AnomalyType.STANDARD
        
        if is_ultra_rare:
            rtp_value = round(random.uniform(107.0, 109.8), 1)
            confidence = random.randint(96, 99)
            duration_minutes = random.randint(45, 75)
            prefix = "ULT"
        else:
            rtp_value = round(random.uniform(102.5, 108.2), 1)
            confidence = random.randint(88, 97)
            duration_minutes = random.randint(75, 140)
            prefix = "STD"
        
        anomaly_code = f"{prefix}-{secrets.token_hex(3).upper()}"
        optimal_bet = round(random.uniform(2.5, 6.0), 1)
        
        # Save anomaly
        async with Database.pool.acquire() as conn:
            anomaly_id = await conn.fetchval("""
                INSERT INTO anomalies (
                    user_id, anomaly_code, type, casino_name, casino_offer_id,
                    slot_name, rtp_value, confidence, volatility, expected_payout_range,
                    optimal_bet, region, expires_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                RETURNING id
            """, user_id, anomaly_code, anomaly_type.value, casino_name, 
                casino_info['offer_id'], slot['name'], rtp_value, confidence,
                slot['volatility'], slot['exp_win'], optimal_bet, region,
                datetime.utcnow() + timedelta(minutes=duration_minutes))
            
            # Update user scans
            await conn.execute("""
                UPDATE users 
                SET scans_remaining = scans_remaining - 1,
                    total_scans = total_scans + 1,
                    last_active_at = NOW()
                WHERE id = $1
            """, user_id)
        
        return {
            'id': str(anomaly_id),
            'anomaly_code': anomaly_code,
            'type': anomaly_type.value,
            'casino_name': casino_name,
            'casino_offer_id': casino_info['offer_id'],
            'slot_name': slot['name'],
            'rtp_value': rtp_value,
            'confidence': confidence,
            'volatility': slot['volatility'],
            'expected_payout_range': slot['exp_win'],
            'optimal_bet': optimal_bet,
            'region': region,
            'expires_at': datetime.utcnow() + timedelta(minutes=duration_minutes)
        }

# =====================================================
# LIFESPAN CONTEXT MANAGER
# =====================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await Database.connect()
    global redis_client
    redis_client = await redis.from_url(REDIS_URL)
    logger.info("✅ SlotGenius API started successfully")
    
    yield
    
    # Shutdown
    await Database.disconnect()
    await redis_client.close()
    logger.info("👋 SlotGenius API shutdown complete")

# =====================================================
# FASTAPI APP
# =====================================================
app = FastAPI(
    title="SlotGenius AI Platform API",
    version="2.0.0",
    description="Advanced RTP anomaly detection platform",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://slotgeniusai.com",
        "https://www.slotgeniusai.com",
        "https://*.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# API ENDPOINTS
# =====================================================

@app.get("/")
async def root():
    return {"message": "SlotGenius AI Platform API", "version": "2.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

# Authentication Endpoints
@app.post("/api/auth/register", status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegister, background_tasks: BackgroundTasks):
    async with Database.pool.acquire() as conn:
        # Check if user exists
        existing = await conn.fetchrow(
            "SELECT id FROM users WHERE email = $1 OR username = $2",
            user_data.email, user_data.username
        )
        
        if existing:
            raise HTTPException(status_code=400, detail="User already exists")
        
        # Hash password
        password_hash = hashlib.sha256(user_data.password.encode()).hexdigest()
        
        # Generate affise clickid
        affise_clickid = f"sg_web_{int(datetime.utcnow().timestamp())}"
        
        # Create user
        user_id = await conn.fetchval("""
            INSERT INTO users (
                email, username, password_hash, 
                utm_source, utm_medium, utm_campaign, utm_content, utm_term,
                fbclid, affise_clickid
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING id
        """, user_data.email, user_data.username, password_hash,
            user_data.utm_source or 'direct', user_data.utm_medium or 'web', 
            user_data.utm_campaign or 'platform', user_data.utm_content, 
            user_data.utm_term, user_data.fbclid, affise_clickid)
        
        # Track Meta event
        background_tasks.add_task(
            MetaPixelService.track_event,
            str(user_id),
            "CompleteRegistration",
            {
                "content_name": "Platform Registration",
                "content_category": "user_registration",
                "utm_source": user_data.utm_source or 'direct'
            }
        )
        
        # Create token
        access_token = await create_access_token({"sub": str(user_id)})
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user_id": str(user_id),
            "username": user_data.username
        }

@app.post("/api/auth/login")
async def login(credentials: UserLogin):
    async with Database.pool.acquire() as conn:
        user = await conn.fetchrow("""
            SELECT id, username, password_hash FROM users WHERE email = $1
        """, credentials.email)
        
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Verify password
        password_hash = hashlib.sha256(credentials.password.encode()).hexdigest()
        if password_hash != user['password_hash']:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Update last login
        await conn.execute(
            "UPDATE users SET last_active_at = $1 WHERE id = $2",
            datetime.utcnow(), user['id']
        )
        
        # Create token
        access_token = await create_access_token({"sub": str(user['id'])})
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user_id": str(user['id']),
            "username": user['username']
        }

# User Endpoints
@app.get("/api/users/profile")
async def get_profile(user_id: str = Depends(verify_token)):
    async with Database.pool.acquire() as conn:
        user = await conn.fetchrow("""
            SELECT id, email, username, subscription_tier, scans_remaining,
                   ultra_rare_eligible, ltv_score, total_scans, total_conversions,
                   total_revenue, created_at
            FROM users WHERE id = $1
        """, user_id)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {
            "id": str(user['id']),
            "email": user['email'],
            "username": user['username'],
            "subscription_tier": user['subscription_tier'],
            "scans_remaining": user['scans_remaining'],
            "ultra_rare_eligible": user['ultra_rare_eligible'],
            "ltv_score": user['ltv_score'],
            "total_scans": user['total_scans'],
            "total_conversions": user['total_conversions'],
            "total_revenue": user['total_revenue'],
            "created_at": user['created_at']
        }

# Anomaly Endpoints
@app.get("/api/anomalies/active")
async def get_active_anomalies(user_id: str = Depends(verify_token)):
    async with Database.pool.acquire() as conn:
        anomalies = await conn.fetch("""
            SELECT * FROM anomalies
            WHERE user_id = $1 AND expires_at > $2 AND claimed_at IS NULL
            ORDER BY created_at DESC
        """, user_id, datetime.utcnow())
        
        return {
            "anomalies": [
                {
                    "id": str(a['id']),
                    "anomaly_code": a['anomaly_code'],
                    "type": a['type'],
                    "casino_name": a['casino_name'],
                    "slot_name": a['slot_name'],
                    "rtp_value": a['rtp_value'],
                    "confidence": a['confidence'],
                    "volatility": a['volatility'],
                    "expected_payout_range": a['expected_payout_range'],
                    "optimal_bet": a['optimal_bet'],
                    "region": a['region'],
                    "expires_at": a['expires_at'],
                    "created_at": a['created_at']
                }
                for a in anomalies
            ]
        }

@app.post("/api/anomalies/scan")
async def scan_anomalies(
    scan_request: AnomalyScanRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(verify_token)
):
    # Check user limits
    async with Database.pool.acquire() as conn:
        user = await conn.fetchrow("""
            SELECT scans_remaining, ultra_rare_eligible, ltv_score
            FROM users WHERE id = $1
        """, user_id)
        
        if not user or user['scans_remaining'] <= 0:
            raise HTTPException(status_code=429, detail="Scan limit reached")
        
        # Check ultra rare permission
        if scan_request.scan_type == 'ultra_rare' and not user['ultra_rare_eligible']:
            raise HTTPException(status_code=403, detail="Ultra Rare scans require premium")
    
    # Perform scans for each region
    anomalies_found = []
    for region in scan_request.regions[:3]:  # Max 3 regions per scan
        anomaly = await AnomalyEngine.create_anomaly(user_id, region, scan_request.scan_type)
        if anomaly:
            anomalies_found.append(anomaly)
    
    # Track Meta event for first scan
    if user['ltv_score'] == 0:
        background_tasks.add_task(
            MetaPixelService.track_event,
            user_id,
            "AddToCart",
            {
                "content_name": "First Anomaly Scan",
                "content_category": "anomaly_scan",
                "scan_type": scan_request.scan_type
            }
        )
    
    # Update LTV score
    async with Database.pool.acquire() as conn:
        await conn.execute("""
            UPDATE users 
            SET ltv_score = LEAST(ltv_score + 5, 100)
            WHERE id = $1
        """, user_id)
    
    return {
        "scans_performed": len(scan_request.regions),
        "anomalies_found": len(anomalies_found),
        "anomalies": anomalies_found
    }

@app.post("/api/anomalies/{anomaly_id}/claim")
async def claim_anomaly(
    anomaly_id: str,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(verify_token)
):
    async with Database.pool.acquire() as conn:
        # Get anomaly
        anomaly = await conn.fetchrow("""
            SELECT * FROM anomalies
            WHERE id = $1 AND user_id = $2 AND expires_at > $3 AND claimed_at IS NULL
        """, anomaly_id, user_id, datetime.utcnow())
        
        if not anomaly:
            raise HTTPException(status_code=404, detail="Anomaly not found or expired")
        
        # Mark as claimed
        await conn.execute("""
            UPDATE anomalies SET claimed_at = $1 WHERE id = $2
        """, datetime.utcnow(), anomaly_id)
        
        # Generate affiliate URL
        affiliate_url = await CasinoRouter.generate_affise_url(
            user_id,
            anomaly['casino_name'],
            anomaly_id
        )
        
        # Save click
        await conn.execute("""
            INSERT INTO casino_clicks (user_id, anomaly_id, casino_name, casino_offer_id, affise_url)
            VALUES ($1, $2, $3, $4, $5)
        """, user_id, anomaly_id, anomaly['casino_name'], anomaly['casino_offer_id'], affiliate_url)
        
        # Track Meta event
        background_tasks.add_task(
            MetaPixelService.track_event,
            user_id,
            "InitiateCheckout",
            {
                "content_name": f"Casino Click - {anomaly['casino_name']}",
                "content_category": "casino_redirect",
                "casino_name": anomaly['casino_name']
            }
        )
        
        return {"redirect_url": affiliate_url}

# Analytics Endpoints
@app.get("/api/analytics/dashboard")
async def get_dashboard(user_id: str = Depends(verify_token)):
    async with Database.pool.acquire() as conn:
        # Get user stats
        stats = await conn.fetchrow("""
            SELECT 
                u.*,
                COUNT(DISTINCT a.id) as anomalies_created,
                COUNT(DISTINCT cc.id) as casino_clicks,
                COUNT(DISTINCT c.id) as conversions_count
            FROM users u
            LEFT JOIN anomalies a ON u.id = a.user_id
            LEFT JOIN casino_clicks cc ON u.id = cc.user_id
            LEFT JOIN conversions c ON u.id = c.user_id
            WHERE u.id = $1
            GROUP BY u.id
        """, user_id)
        
        # Get recent anomalies
        recent_anomalies = await conn.fetch("""
            SELECT * FROM anomalies
            WHERE user_id = $1
            ORDER BY created_at DESC
            LIMIT 5
        """, user_id)
        
        return {
            "stats": {
                "scans_remaining": stats['scans_remaining'],
                "total_scans": stats['total_scans'],
                "anomalies_created": stats['anomalies_created'],
                "casino_clicks": stats['casino_clicks'],
                "conversions": stats['conversions_count'],
                "total_revenue": stats['total_revenue'],
                "ultra_rare_eligible": stats['ultra_rare_eligible']
            },
            "recent_anomalies": [
                {
                    "id": str(a['id']),
                    "anomaly_code": a['anomaly_code'],
                    "type": a['type'],
                    "casino_name": a['casino_name'],
                    "slot_name": a['slot_name'],
                    "created_at": a['created_at']
                }
                for a in recent_anomalies
            ]
        }

# Webhook Endpoints
@app.post("/webhook/postback")
async def handle_postback(
    request: Request,
    background_tasks: BackgroundTasks
):
    """Handle conversion postbacks from casinos"""
    # Get parameters from query string or form data
    params = dict(request.query_params)
    
    # Extract required parameters
    user_id = params.get('user_id')
    conversion_type = params.get('type')
    offer_id = params.get('offer_id')
    click_id = params.get('click_id')
    amount = params.get('amount')
    
    if not all([user_id, conversion_type, offer_id]):
        return {"status": "error", "message": "Missing required parameters"}
    
    # Find casino by offer_id
    casino_name = None
    commission = 0
    for name, data in CasinoRouter.CASINO_OFFERS.items():
        if data['offer_id'] == offer_id:
            casino_name = name
            commission = data['commission']
            break
    
    if not casino_name:
        return {"status": "error", "message": "Unknown offer_id"}
    
    try:
        async with Database.pool.acquire() as conn:
            # Save conversion
            await conn.execute("""
                INSERT INTO conversions (
                    user_id, conversion_type, casino_name, offer_id, 
                    conversion_value, user_deposit_amount, postback_data
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, user_id, conversion_type, casino_name, offer_id, 
                commission if conversion_type == 'deposit' else 0,
                float(amount) if amount else None,
                json.dumps(params))
            
            # Update user stats
            if conversion_type == 'deposit':
                await conn.execute("""
                    UPDATE users 
                    SET total_conversions = total_conversions + 1,
                        total_revenue = total_revenue + $1,
                        scans_remaining = scans_remaining + 3
                    WHERE id = $2
                """, commission, user_id)
        
        # Track Meta events
        if conversion_type == 'registration':
            background_tasks.add_task(
                MetaPixelService.track_event,
                user_id,
                "CompleteRegistration",
                {
                    "content_name": f"Casino Registration - {casino_name}",
                    "content_category": "casino_registration"
                }
            )
        elif conversion_type == 'deposit':
            background_tasks.add_task(
                MetaPixelService.track_event,
                user_id,
                "Purchase",
                {
                    "content_name": f"Casino Deposit - {casino_name}",
                    "value": commission,
                    "currency": "EUR"
                }
            )
        
        return {"status": "success", "message": f"Conversion tracked: {conversion_type}"}
        
    except Exception as e:
        logger.error(f"Postback error: {e}")
        return {"status": "error", "message": str(e)}

# Test endpoint for Railway deployment
@app.get("/test")
async def test():
    return {
        "status": "ok",
        "database": "connected" if Database.pool else "not connected",
        "redis": "connected" if redis_client else "not connected",
        "timestamp": datetime.utcnow()
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
