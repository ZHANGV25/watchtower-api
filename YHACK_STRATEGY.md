# YHack 2026 - WatchTower Strategy

## Track Submissions (Priority Order)

### 1. Societal Impact - Healthcare (ASUS)
**Angle**: Elder care fall detection + caregiver alerting
**Why it fits**: Falls are #1 injury death cause for 65+. WatchTower turns any cheap camera into a safety monitor without wearables.
**What to demo**: Elderly person falls -> system detects lying pose -> LLM verifies -> alert fires with captured frame -> caregiver notified
**Needs**: Polish fall detection demo, add SMS/notification mockup, prepare human story for pitch

### 2. Personal AI Agent (Harper)
**Angle**: Autonomous monitoring agent that reasons about scenes
**Why it fits**: NL instructions -> autonomous detection + LLM verification loop -> action-taking
**What to demo**: User describes scenario ("elderly mother home alone") -> agent generates full monitoring plan (multiple rules, zones, severity levels, time windows) -> runs autonomously
**Needs**: Build agentic planning feature (LLM generates multi-rule monitoring plans from a scenario description)

### 3. Best Use of Viam
**Angle**: Viam-managed Pi camera with ML vision pipeline
**Why it fits**: Viam criteria = "vision + data capture + ML" which is exactly WatchTower
**What to demo**: Pi camera controlled through Viam -> feeds into WatchTower detection -> data capture for replay
**Needs**: Visit Viam booth Saturday morning, get a Pi, integrate Viam SDK as camera source
**Note**: They are lending Raspberry Pis and robotic arms at the event

### 4. Best Use of K2 Think V2 (MBZUAI IFM)
**Angle**: Swap rule parsing or verification LLM to K2
**Why it fits**: 70B reasoning model for complex rule compilation
**What to demo**: Show K2 handling multi-step rule reasoning
**Needs**: Get API access, swap one LLM call, test it works
**Prize**: reMarkable tablet per team member

### 5. Best UI/UX
**Angle**: Command center aesthetic, dense information design
**Free submission - no extra work needed**

### 6. Grand Prize
**Free submission - no extra work needed**

---

## Pre-Hackathon TODO

### Must Do (Before Saturday 11AM)
- [ ] Prepare slide deck (see Pitch Structure below)
- [ ] Record 30-second demo video of current system working
- [ ] Test fall detection scenario end-to-end (person lies down -> alert fires)
- [ ] Prepare the "elderly care" story/narrative for pitch
- [ ] Ensure both repos are clean and ready (they will check GitHub)

### Build During Hackathon (Priority Order)
1. **Agentic monitoring plans** (2-3 hours)
   - New endpoint: user describes a scenario in 1-2 sentences
   - LLM generates a full monitoring plan: multiple rules + zones + time windows
   - Frontend: "Create monitoring plan" button that opens a scenario input
   - This is the key differentiator for the AI Agent track

2. **SMS/Push notification on alert** (1-2 hours)
   - Integrate Twilio or similar for SMS alerts
   - Shows the system takes real action, not just UI alerts
   - Critical for both Societal Impact and AI Agent tracks

3. **Viam integration** (2-3 hours, at event)
   - Get Pi from Viam booth
   - Use Viam SDK to capture frames instead of local OpenCV
   - Keep rest of pipeline the same

4. **K2 Think V2 swap** (1 hour, if API available)
   - Swap rule_parser.py to use K2 API
   - Keep Haiku for verification (speed matters there)

5. **Demo polish** (ongoing)
   - Ensure fall detection works reliably from a room-distance camera
   - Have 2-3 pre-built rules ready to show
   - Clean up any remaining UI issues

---

## Pitch Structure (3-5 minute presentation)

### Slide 1: The Problem (30 sec)
- "X million elderly people live alone in the US"
- "Falls are the leading cause of injury death for adults 65+"
- "Current solutions: $30/month wearable pendants that seniors forget to wear"
- "What if any $20 camera could be a safety monitor?"

### Slide 2: The Solution (30 sec)
- "WatchTower: describe what to watch for in plain English"
- "An AI agent translates your words into real-time detection rules"
- Show the UI with a rule being typed: "Alert if someone falls down"

### Slide 3: How It Works (45 sec)
- Architecture diagram: Camera -> YOLO Detection -> Rule Engine -> LLM Verification -> Alert
- Key insight: "YOLO detects, the LLM thinks. False positives are filtered by vision AI before alerts fire."
- "Natural language in, verified alerts out"

### Slide 4: Live Demo (60-90 sec)
- Show the live camera feed with detections
- Type a rule: "Alert if a person is detected in the kitchen"
- Show auto-zone detection
- Trigger an alert, show the LLM verification
- Show the replay system
- If Viam: show the Pi camera unit

### Slide 5: Agentic Monitoring (30 sec)
- "Describe a scenario, the agent builds a complete safety plan"
- Type: "My elderly mother is home alone"
- Agent generates: fall detection rule, inactivity alert, front door monitoring, nighttime-only kitchen alerts
- "One sentence in, comprehensive monitoring out"

### Slide 6: Impact & Future (30 sec)
- Elder care, child safety, small business security
- Multi-camera support, mobile app, RTSP/IP camera integration
- "Any camera. Any rule. Plain English."

### Slide 7: Tech Stack (15 sec, if time)
- Next.js + FastAPI + YOLO v8 + MediaPipe + Claude (Bedrock) + WebSocket
- Keep brief, judges care about what it does not what it's built with

---

## Demo Day Checklist (Sunday)

- [ ] Laptop running watchtower-api with camera
- [ ] Second screen or projector showing watchtower-web
- [ ] Pre-created zones and 2-3 rules ready
- [ ] Fall detection scenario rehearsed
- [ ] Agentic planning demo rehearsed
- [ ] Slide deck loaded and ready
- [ ] 30-second video submitted by 11:30 AM
- [ ] If Viam: Pi camera unit set up and streaming

---

## Key Talking Points for Judges

**For AI Agent judges (Harper)**:
- "The system autonomously reasons about what it sees - it doesn't just pattern match"
- "LLM verification eliminates false positives that plague traditional systems"
- "One natural language instruction generates a complete monitoring workflow"
- "This is an agent, not a tool - it plans, monitors, verifies, and acts"

**For Societal Impact judges (ASUS)**:
- "Falls kill 36,000 Americans over 65 every year"
- "This works with any existing camera - no special hardware or wearables"
- "Caregivers get peace of mind with real-time verified alerts"
- "The natural language interface means anyone can set it up, not just tech-savvy users"

**For Viam judges**:
- "We use Viam as the hardware abstraction layer for camera management"
- "Vision + data capture + ML - all three pillars of the Viam platform"
- "The Pi unit could be deployed in any room and managed remotely through Viam"
