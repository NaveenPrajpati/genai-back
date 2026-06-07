-- Idempotent meal logging.
--
-- The app upserts meal_slots with on_conflict="plan_id,day_of_week,meal_type".
-- Postgres ON CONFLICT only dedupes when a matching unique constraint exists,
-- so this migration (1) removes any pre-existing duplicate slots, keeping one
-- arbitrary row per (plan_id, day_of_week, meal_type), then (2) adds the
-- constraint. Run this once against the Supabase Postgres database.

DELETE FROM meal_slots a
USING meal_slots b
WHERE a.ctid < b.ctid
  AND a.plan_id = b.plan_id
  AND a.day_of_week = b.day_of_week
  AND a.meal_type = b.meal_type;

ALTER TABLE meal_slots
  ADD CONSTRAINT meal_slots_plan_day_meal_unique
  UNIQUE (plan_id, day_of_week, meal_type);
