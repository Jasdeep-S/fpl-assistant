DROP TABLE IF EXISTS features;

CREATE TABLE features AS
WITH team_matches AS (
    -- one row per team per match, with goals for and against
    SELECT
        season, team, kickoff_time, opponent_team, was_home,
        MAX(CASE WHEN was_home THEN team_h_score ELSE team_a_score END) AS goals_for,
        MAX(CASE WHEN was_home THEN team_a_score ELSE team_h_score END) AS goals_against
    FROM raw_history_seasons
    GROUP BY season, team, kickoff_time, opponent_team, was_home
),
team_form AS (
    -- each team's rolling form BEFORE each match
    SELECT
        season, team, kickoff_time,
        AVG(goals_for)                  OVER tw AS gf_last5,
        AVG(goals_against)              OVER tw AS ga_last5,
        AVG((goals_against = 0)::int)   OVER tw AS cs_rate_last5
    FROM team_matches
    WINDOW tw AS (PARTITION BY season, team ORDER BY kickoff_time
                  ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING)
)
SELECT
    h.season,
    h.element AS player_id,
    h.name,
    h.position,
    h.team,
    h."GW" AS gameweek,
    h.kickoff_time,
    h.was_home,
    h.value AS price,

    -- TARGET
    h.total_points AS target_points,

    -- player features (previous matches only)
    AVG(h.total_points) OVER w5 AS avg_pts_last5,
    AVG(h.total_points) OVER w3 AS avg_pts_last3,
    AVG(h.minutes)      OVER w5 AS avg_minutes_last5,
    AVG(h.goals_scored) OVER w5 AS avg_goals_last5,
    AVG(h.assists)      OVER w5 AS avg_assists_last5,
    AVG(h.bonus)        OVER w5 AS avg_bonus_last5,
    AVG(h.bps)          OVER w5 AS avg_bps_last5,
    AVG(h.ict_index::float) OVER w5 AS avg_ict_last5,
    COUNT(*)            OVER w5 AS games_in_window,

    -- own team features
    tf.gf_last5      AS team_gf_last5,
    tf.ga_last5      AS team_ga_last5,
    tf.cs_rate_last5 AS team_cs_rate_last5,

    -- opponent features
    opp.gf_last5      AS opp_gf_last5,
    opp.ga_last5      AS opp_ga_last5,
    opp.cs_rate_last5 AS opp_cs_rate_last5
FROM raw_history_seasons h
JOIN raw_teams_seasons ot
  ON ot.season = h.season AND ot.id = h.opponent_team
LEFT JOIN team_form tf
  ON tf.season = h.season AND tf.team = h.team
 AND tf.kickoff_time = h.kickoff_time
LEFT JOIN team_form opp
  ON opp.season = h.season AND opp.team = ot.name
 AND opp.kickoff_time = h.kickoff_time
WHERE h.position IN ('GK', 'DEF', 'MID', 'FWD')
WINDOW
    w5 AS (PARTITION BY h.season, h.element ORDER BY h.kickoff_time
           ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING),
    w3 AS (PARTITION BY h.season, h.element ORDER BY h.kickoff_time
           ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING);