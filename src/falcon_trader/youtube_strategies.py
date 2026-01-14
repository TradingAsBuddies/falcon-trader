#!/usr/bin/env python3
"""
YouTube Trading Strategy Collection System
Extracts trading strategies from YouTube videos using AI analysis
"""

import datetime
import json
import os
import re
from typing import Dict, List, Optional
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
import anthropic

from falcon_core import get_db_manager


class YouTubeStrategyDB:
    """Database manager for YouTube trading strategies using falcon_core DatabaseManager"""

    def __init__(self, db_path=None, db_manager=None):
        """Initialize with optional db_path (legacy) or db_manager"""
        self.db = db_manager or get_db_manager()
        print(f"[DB] YouTube strategies using {self.db.db_type} database")

    def _parse_json(self, value):
        """Parse JSON string or return value if already parsed"""
        if value is None:
            return []
        if isinstance(value, (list, dict)):
            return value
        try:
            return json.loads(value) if value else []
        except (json.JSONDecodeError, TypeError):
            return []

    def add_strategy(self, strategy_data: Dict) -> int:
        """Add a new strategy to the database"""
        timestamp = datetime.datetime.now()
        if self.db.db_type == 'sqlite':
            timestamp = timestamp.isoformat()

        sql = '''
            INSERT INTO youtube_strategies (
                title, creator, youtube_url, video_id, description,
                strategy_overview, trading_style, instruments,
                entry_rules, exit_rules, risk_management,
                strategy_code, tags, performance_metrics,
                pros, cons, created_at, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        '''
        params = (
            strategy_data.get('title', ''),
            strategy_data.get('creator', ''),
            strategy_data.get('youtube_url', ''),
            strategy_data.get('video_id', ''),
            strategy_data.get('description', ''),
            strategy_data.get('strategy_overview', ''),
            strategy_data.get('trading_style', ''),
            strategy_data.get('instruments', ''),
            strategy_data.get('entry_rules', ''),
            strategy_data.get('exit_rules', ''),
            strategy_data.get('risk_management', ''),
            strategy_data.get('strategy_code', ''),
            json.dumps(strategy_data.get('tags', [])),
            json.dumps(strategy_data.get('performance_metrics', {})),
            strategy_data.get('pros', ''),
            strategy_data.get('cons', ''),
            timestamp,
            timestamp
        )

        return self.db.execute(sql, params)

    def get_all_strategies(self) -> List[Dict]:
        """Get all strategies from database"""
        sql = 'SELECT * FROM youtube_strategies ORDER BY created_at DESC'
        rows = self.db.execute(sql, fetch='all')

        strategies = []
        for row in rows:
            strategy = dict(row) if hasattr(row, 'keys') else dict(zip(
                ['id', 'title', 'creator', 'youtube_url', 'video_id', 'description',
                 'strategy_overview', 'trading_style', 'instruments', 'entry_rules',
                 'exit_rules', 'risk_management', 'strategy_code', 'tags',
                 'performance_metrics', 'pros', 'cons', 'created_at', 'updated_at'],
                row
            ))
            strategy['tags'] = self._parse_json(strategy.get('tags'))
            strategy['performance_metrics'] = self._parse_json(strategy.get('performance_metrics'))
            strategies.append(strategy)

        return strategies

    def get_strategy_by_id(self, strategy_id: int) -> Optional[Dict]:
        """Get a specific strategy by ID"""
        sql = 'SELECT * FROM youtube_strategies WHERE id = %s'
        row = self.db.execute(sql, (strategy_id,), fetch='one')

        if row:
            strategy = dict(row) if hasattr(row, 'keys') else dict(zip(
                ['id', 'title', 'creator', 'youtube_url', 'video_id', 'description',
                 'strategy_overview', 'trading_style', 'instruments', 'entry_rules',
                 'exit_rules', 'risk_management', 'strategy_code', 'tags',
                 'performance_metrics', 'pros', 'cons', 'created_at', 'updated_at'],
                row
            ))
            strategy['tags'] = self._parse_json(strategy.get('tags'))
            strategy['performance_metrics'] = self._parse_json(strategy.get('performance_metrics'))
            return strategy

        return None


class YouTubeStrategyExtractor:
    """Extract trading strategies from YouTube videos using AI"""

    def __init__(self, claude_api_key: Optional[str] = None):
        self.claude_api_key = claude_api_key or os.getenv('CLAUDE_API_KEY')
        if self.claude_api_key:
            self.client = anthropic.Anthropic(api_key=self.claude_api_key)
        else:
            self.client = None
            print("[WARNING] No Claude API key found. Strategy extraction will be limited.")

    def extract_video_id(self, youtube_url: str) -> str:
        """Extract video ID from YouTube URL"""
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/)([^&\n?#]+)',
            r'youtube\.com/embed/([^&\n?#]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, youtube_url)
            if match:
                return match.group(1)

        raise ValueError("Invalid YouTube URL")

    def get_transcript(self, video_id: str) -> str:
        """Get YouTube video transcript"""
        try:
            api = YouTubeTranscriptApi()
            result = api.fetch(video_id)
            transcript = ' '.join([snippet.text for snippet in result.snippets])
            return transcript
        except (TranscriptsDisabled, NoTranscriptFound) as e:
            raise Exception(f"Could not fetch transcript: {str(e)}")

    def extract_strategy_with_ai(self, transcript: str, video_title: str = "") -> Dict:
        """Use Claude to extract trading strategy from transcript (fabric-style)"""

        if not self.client:
            return {"error": "Claude API key not configured"}

        # Fabric-style extract_wisdom prompt
        prompt = f"""You are analyzing a YouTube video about trading strategies. Extract the key trading strategy wisdom from this transcript.

Video Title: {video_title}

Transcript:
{transcript[:15000]}  # Limit to avoid token overflow

Please extract and structure the following information:

1. STRATEGY NAME (a concise, descriptive name for the strategy - NOT the video title, e.g., "Universal Liquidity Playbook", "3-Bar Reversal Strategy")
2. PRESENTER (the person presenting/teaching the strategy if mentioned - name or alias, e.g., "Zee", "Ross Cameron")
3. STRATEGY OVERVIEW (2-3 sentences describing the core approach)
4. TRADING STYLE (day trading, swing trading, scalping, etc.)
5. INSTRUMENTS (stocks, options, futures, forex, crypto)
6. ENTRY RULES (specific conditions for entering trades)
7. EXIT RULES (specific conditions for exiting trades, profit targets, stop losses)
8. RISK MANAGEMENT (position sizing, stop loss strategy, risk per trade)
9. STRATEGY CODE (if any Python/Pine Script code is mentioned, extract it)
10. TAGS (list 3-5 relevant tags: momentum, mean-reversion, breakout, etc.)
11. PERFORMANCE METRICS (any mentioned win rate, profit factor, drawdown, etc.)
12. PROS (advantages of this strategy)
13. CONS (limitations or risks of this strategy)

Format your response as JSON with these exact keys:
{{
  "strategy_name": "Brief descriptive name",
  "presenter": "Name or alias if mentioned, empty string if not",
  "strategy_overview": "...",
  "trading_style": "...",
  "instruments": "...",
  "entry_rules": "...",
  "exit_rules": "...",
  "risk_management": "...",
  "strategy_code": "...",
  "tags": ["tag1", "tag2", "tag3"],
  "performance_metrics": {{"win_rate": "...", "profit_factor": "..."}},
  "pros": "...",
  "cons": "..."
}}

Be concise and specific. Extract only factual information from the transcript."""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=4000,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            content = response.content[0].text

            # Try to extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                strategy_data = json.loads(json_match.group())
                return strategy_data
            else:
                return {"error": "Could not parse AI response as JSON", "raw_response": content}

        except Exception as e:
            return {"error": f"AI extraction failed: {str(e)}"}

    def get_video_metadata(self, video_id: str) -> Dict:
        """Get video title and channel name using yt-dlp"""
        try:
            import yt_dlp
            ydl_opts = {'quiet': True, 'no_warnings': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
                return {
                    'title': info.get('title', f'Strategy from {video_id}'),
                    'channel': info.get('channel', 'Unknown Creator')
                }
        except Exception as e:
            print(f"[WARNING] Could not fetch video metadata: {e}")
            return {
                'title': f'Strategy from {video_id}',
                'channel': 'Unknown Creator'
            }

    def process_youtube_url(self, youtube_url: str) -> Dict:
        """Complete pipeline: URL -> Transcript -> AI Extraction -> Structured Data"""

        try:
            # Extract video ID
            video_id = self.extract_video_id(youtube_url)
            print(f"[INFO] Extracted video ID: {video_id}")

            # Get video metadata
            print("[INFO] Fetching video metadata...")
            metadata = self.get_video_metadata(video_id)
            video_title = metadata['title']
            creator = metadata['channel']
            print(f"[INFO] Title: {video_title}")
            print(f"[INFO] Creator: {creator}")

            # Get transcript
            print("[INFO] Fetching transcript...")
            transcript = self.get_transcript(video_id)
            print(f"[INFO] Transcript length: {len(transcript)} characters")

            # Extract strategy using AI
            print("[INFO] Analyzing with Claude AI...")
            strategy_data = self.extract_strategy_with_ai(transcript, video_title)

            if "error" in strategy_data:
                return strategy_data

            # Create a meaningful title
            strategy_name = strategy_data.get('strategy_name', 'Trading Strategy')
            presenter = strategy_data.get('presenter', '')

            if presenter:
                # Format: "Strategy Name by Presenter (Channel)"
                title = f"{strategy_name} by {presenter}"
                if creator and creator.lower() not in presenter.lower():
                    title += f" ({creator})"
                creator_field = f"{presenter} - {creator}" if creator != presenter else presenter
            else:
                # Format: "Strategy Name (Channel)"
                title = f"{strategy_name} ({creator})" if creator else strategy_name
                creator_field = creator

            # Add metadata
            strategy_data['title'] = title
            strategy_data['creator'] = creator_field
            strategy_data['youtube_url'] = youtube_url
            strategy_data['video_id'] = video_id
            strategy_data['description'] = f"Trading strategy extracted from YouTube video: {video_title}"

            return strategy_data

        except Exception as e:
            return {"error": str(e)}


def main():
    """CLI interface for YouTube strategy extraction"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 youtube_strategies.py <youtube_url>")
        print("Example: python3 youtube_strategies.py https://www.youtube.com/watch?v=VIDEO_ID")
        sys.exit(1)

    youtube_url = sys.argv[1]

    # Initialize
    db = YouTubeStrategyDB()
    extractor = YouTubeStrategyExtractor()

    # Process video
    print(f"[START] Processing YouTube URL: {youtube_url}")
    strategy_data = extractor.process_youtube_url(youtube_url)

    if "error" in strategy_data:
        print(f"[ERROR] {strategy_data['error']}")
        sys.exit(1)

    # Save to database
    print("[INFO] Saving to database...")
    strategy_id = db.add_strategy(strategy_data)

    print(f"\n[SUCCESS] Strategy saved with ID: {strategy_id}")
    print(f"Title: {strategy_data.get('title', 'N/A')}")
    print(f"Trading Style: {strategy_data.get('trading_style', 'N/A')}")
    print(f"Tags: {', '.join(strategy_data.get('tags', []))}")


if __name__ == '__main__':
    main()
