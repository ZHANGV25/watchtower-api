"""Seed the local database with realistic elder care demo data.

Run: python seed_demo.py
Creates a room with activity entries, alerts, and memory entries
so the dashboard has data to display immediately.
"""
import asyncio
import os
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("WATCHTOWER_DB_BACKEND", "sqlite")
os.environ.setdefault("WATCHTOWER_DB", "./data/watchtower.db")

import database as db
from models import Alert, Camera, Condition, MemoryEntry, Rule, User
from auth import hash_password


async def seed():
    await db.init_db()

    # Ensure demo user
    user = await db.get_user_by_username("vhz2020@gmail.com")
    if not user:
        await db.create_user(User(username="vhz2020@gmail.com", password_hash=hash_password("12734836")))
        print("Created demo user")

    # Clear existing data
    existing = await db.list_cameras()
    for cam in existing:
        await db.delete_camera(cam.id)
    print("Cleared existing cameras")

    # Create rooms
    living_room = Camera(name="Mom's Living Room", location="Downstairs", status="online")
    living_room.last_seen = time.time() - 120
    await db.create_camera(living_room)

    kitchen = Camera(name="Kitchen", location="Downstairs", status="online")
    kitchen.last_seen = time.time() - 300
    await db.create_camera(kitchen)

    bedroom = Camera(name="Mom's Bedroom", location="Upstairs", status="online")
    bedroom.last_seen = time.time() - 600
    await db.create_camera(bedroom)

    print(f"Created rooms: {living_room.id}, {kitchen.id}, {bedroom.id}")

    # Add elder care rules for each room
    from routes.cameras import _create_elder_care_rules
    for cam in [living_room, kitchen, bedroom]:
        await _create_elder_care_rules(cam.id)
    print("Created elder care rules for all rooms")

    # Add a custom concern rule to living room
    custom_rule = Rule(
        camera_id=living_room.id,
        name="Hydration reminder",
        natural_language="Mom forgets to drink water - watch how often she goes to the kitchen",
        conditions=[
            Condition(type="object_absent", params={"class": "cup"}),
            Condition(type="duration", params={"seconds": 7200}),
            Condition(type="time_window", params={"start_hour": 8, "end_hour": 20}),
        ],
        severity="medium",
        enabled=True,
    )
    await db.create_rule(custom_rule)

    # Add medication rules
    med1 = Rule(
        camera_id=living_room.id,
        name="MED: Blood Pressure Pill",
        natural_language="Take Blood Pressure Pill - Take with breakfast",
        conditions=[
            Condition(type="time_window", params={"start_hour": 8, "end_hour": 9}),
            Condition(type="object_absent", params={"class": "person"}),
        ],
        severity="high",
        enabled=True,
    )
    await db.create_rule(med1)

    med2 = Rule(
        camera_id=living_room.id,
        name="MED: Vitamin D",
        natural_language="Take Vitamin D - After lunch",
        conditions=[
            Condition(type="time_window", params={"start_hour": 13, "end_hour": 14}),
            Condition(type="object_absent", params={"class": "person"}),
        ],
        severity="high",
        enabled=True,
    )
    await db.create_rule(med2)
    print("Created custom concerns and medication rules")

    # --- Seed activity timeline (memory entries) for today ---
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # Timeline is spatially consistent: Mom is in exactly ONE room at a time.
    # Transitions show her leaving one room before appearing in another.
    #
    # 6:25-6:30  Bedroom (waking up)
    # 6:35-7:10  Kitchen (breakfast)
    # 7:15-8:15  Living Room (TV, coffee)
    # 8:20-8:40  Kitchen (water, snack)
    # 8:45-11:00 Living Room (reading, nap)
    # 11:05-11:25 Kitchen (prep)
    # 11:30-12:00 Living Room (resting)
    # 12:05-12:25 Kitchen (lunch prep)
    # 12:30-15:35 Living Room (lunch, visitor, reading)

    living_room_activities = [
        (7, 15, "Person entered the living room from the hallway. Seated on the couch."),
        (7, 20, "Person seated on couch, appears to be watching TV. Cup visible on side table."),
        (7, 45, "Person still on couch watching TV. Relaxed posture."),
        (8, 0, "Person reached for cup on side table. Drinking."),
        (8, 15, "Person standing up from couch. Moving toward the hallway. Person left the room."),
        (8, 45, "Person returned to living room. Seated on couch with a book."),
        (9, 0, "Person reading on the couch. Calm and relaxed."),
        (9, 30, "Person still reading. No changes."),
        (10, 0, "Person appears to be napping on the couch. Eyes closed, relaxed posture."),
        (10, 30, "Person still napping on couch. Breathing appears normal."),
        (11, 0, "Person woke up from nap. Stretching. Getting up from couch. Person left the room toward kitchen."),
        (11, 30, "Person returned to living room with a glass. Seated on couch."),
        (11, 45, "Person watching TV. Glass on side table."),
        (12, 0, "Person getting up from couch. Moving toward kitchen area. Person left the room."),
        (12, 30, "Person returned with a plate. Eating lunch on the couch."),
        (12, 50, "Person finished eating. Plate on side table. Watching TV."),
        (13, 15, "Person seated on couch. Appears comfortable."),
        (13, 30, "Visitor arrived. Two people in the room. Person appears engaged in conversation."),
        (13, 45, "Two people conversing. Person is animated, gesturing while talking."),
        (14, 0, "Visitor and person still talking. Cup of tea visible."),
        (14, 15, "Visitor getting up. Person waving. Visitor leaving."),
        (14, 20, "Person alone again. Seated on couch. Watching TV."),
        (14, 45, "Person reading on couch. Quiet afternoon."),
        (15, 0, "Person still reading. Good posture in chair."),
        (15, 30, "Person got up to use the bathroom. Walking steadily."),
        (15, 35, "Person returned. Seated on couch."),
    ]

    kitchen_activities = [
        (6, 35, "Person entered kitchen from hallway. Walking to the counter."),
        (6, 45, "Person at the counter preparing something. Kettle visible."),
        (6, 55, "Person seated at the kitchen table. Cup and plate visible. Eating breakfast."),
        (7, 10, "Person finishing breakfast. Standing up from table. Person left the kitchen."),
        (8, 20, "Person entered kitchen. Getting a glass of water from the sink."),
        (8, 30, "Person getting a snack from the cabinet."),
        (8, 40, "Person left the kitchen."),
        (11, 5, "Person entered kitchen slowly using walker. Going to the refrigerator."),
        (11, 15, "Person preparing something at the counter. Moving carefully."),
        (11, 25, "Person getting a glass of water. Person left the kitchen."),
        (12, 5, "Person entered kitchen. Preparing lunch. Using microwave."),
        (12, 25, "Person plating food. Person left the kitchen with plate."),
    ]

    bedroom_activities = [
        (6, 25, "Person getting out of bed. Slow movements. Sitting on edge of bed."),
        (6, 28, "Person stood up from bed. Walking toward bathroom."),
        (6, 32, "Person left the bedroom toward the hallway."),
        (22, 0, "Person entered bedroom. Getting ready for bed."),
        (22, 15, "Person in bed. Lights dimmed."),
        (22, 20, "Person appears to be settling in for sleep."),
    ]

    # Add yesterday's bedtime entries too
    yesterday = today - timedelta(days=1)
    yesterday_bedroom = [
        (22, 30, "Person got into bed. Reading a book with bedside lamp on."),
        (22, 45, "Person still reading in bed."),
        (23, 0, "Person turned off lamp. Settling in to sleep."),
    ]

    entry_count = 0
    for hour, minute, summary in living_room_activities:
        ts = today.replace(hour=hour, minute=minute).timestamp()
        if ts < time.time():  # Only add past entries
            entry = MemoryEntry(timestamp=ts, summary=summary, detection_count=1)
            await db.create_memory_entry(living_room.id, entry)
            entry_count += 1

    for hour, minute, summary in kitchen_activities:
        ts = today.replace(hour=hour, minute=minute).timestamp()
        if ts < time.time():
            entry = MemoryEntry(timestamp=ts, summary=summary, detection_count=1)
            await db.create_memory_entry(kitchen.id, entry)
            entry_count += 1

    for hour, minute, summary in bedroom_activities:
        ts = today.replace(hour=hour, minute=minute).timestamp()
        if ts < time.time():
            entry = MemoryEntry(timestamp=ts, summary=summary, detection_count=1)
            await db.create_memory_entry(bedroom.id, entry)
            entry_count += 1

    for hour, minute, summary in yesterday_bedroom:
        ts = yesterday.replace(hour=hour, minute=minute).timestamp()
        entry = MemoryEntry(timestamp=ts, summary=summary, detection_count=1)
        await db.create_memory_entry(bedroom.id, entry)
        entry_count += 1

    print(f"Created {entry_count} activity entries across all rooms")

    # --- Seed alerts ---
    # Visitor detection (low) - living room
    visitor_alert = Alert(
        camera_id=living_room.id,
        rule_id="visitor-1",
        rule_name="Visitor Detection",
        severity="low",
        timestamp=today.replace(hour=13, minute=30).timestamp(),
        narration="A visitor has arrived. Two people are now in the living room, engaged in conversation.",
    )
    if visitor_alert.timestamp < time.time():
        await db.create_alert(visitor_alert)

    # Night wandering (medium) - from last night
    night_alert = Alert(
        camera_id=bedroom.id,
        rule_id="night-1",
        rule_name="Night Wandering",
        severity="medium",
        timestamp=(yesterday.replace(hour=2, minute=15)).timestamp(),
        narration="Person detected moving in the bedroom at 2:15 AM. Brief bathroom visit, returned to bed after 5 minutes.",
    )
    await db.create_alert(night_alert)

    # Inactivity warning (high) - kitchen, yesterday
    inactivity_alert = Alert(
        camera_id=kitchen.id,
        rule_id="inactivity-1",
        rule_name="Inactivity Alert",
        severity="high",
        timestamp=(yesterday.replace(hour=15, minute=0)).timestamp(),
        narration="No activity detected in the kitchen for over 3 hours during the afternoon. Person was last seen leaving the kitchen at 12:20 PM.",
    )
    await db.create_alert(inactivity_alert)

    print("Created demo alerts")

    # --- Add yesterday activity for reports ---
    yesterday_living = [
        (7, 0, "Person woke up and moved to the couch. Turned on TV."),
        (7, 30, "Person watching TV on couch."),
        (8, 0, "Person went to kitchen for breakfast."),
        (8, 30, "Person returned. Reading on couch."),
        (9, 0, "Person reading quietly."),
        (10, 0, "Person napping on couch."),
        (11, 0, "Person woke up. Went to kitchen."),
        (12, 0, "Person eating lunch on couch."),
        (13, 0, "Person watching TV."),
        (14, 0, "Person reading."),
        (15, 0, "Person went to bathroom."),
        (16, 0, "Person back on couch. Watching TV."),
        (17, 0, "Person in living room."),
        (18, 0, "Person eating dinner on couch."),
        (19, 0, "Person watching TV."),
        (20, 0, "Person getting ready for bed."),
    ]

    for hour, minute, summary in yesterday_living:
        ts = yesterday.replace(hour=hour, minute=minute).timestamp()
        entry = MemoryEntry(timestamp=ts, summary=summary, detection_count=1)
        await db.create_memory_entry(living_room.id, entry)
        entry_count += 1

    print(f"Seeded yesterday's data for weekly reports")

    await db.close_db()
    print("\nDone! Refresh the dashboard to see the data.")
    print(f"  Living Room: {living_room.id}")
    print(f"  Kitchen:     {kitchen.id}")
    print(f"  Bedroom:     {bedroom.id}")


if __name__ == "__main__":
    asyncio.run(seed())
