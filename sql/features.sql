DROP TABLE IF EXISTS features;

CREATE TABLE features AS
SELECT
    season,
    element AS player_id,
    name,
    position,
    team,
    "GW" AS gameweek,
    kickoff_time,
    opponent_team,
    was_home,
    value AS price,

    --what we want to predict
    total_points AS target_points,

    -- only use rows BEFORE the current match
    AVG(total_points) OVER w5 AS avg_pts_last5,
    AVG(total_points) OVER w3 AS avg_pts_last3,
    AVG(minutes)      OVER w5 AS avg_minutes_last5,
    AVG(goals_scored) OVER w5 AS avg_goals_last5,
    AVG(assists)      OVER w5 AS avg_assists_last5,
    AVG(bonus)        OVER w5 AS avg_bonus_last5,
    AVG(bps)          OVER w5 AS avg_bps_last5,
    AVG(ict_index::float) OVER w5 AS avg_ict_last5,
    COUNT(*)          OVER w5 AS games_in_window
FROM raw_history_seasons
WHERE position IN ('GK', 'DEF', 'MID', 'FWD')
WINDOW
    w5 AS (PARTITION BY season, element ORDER BY kickoff_time
           ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING),
    w3 AS (PARTITION BY season, element ORDER BY kickoff_time
           ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING);