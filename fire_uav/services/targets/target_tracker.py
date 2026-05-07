from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
# костыль, так как jetson не работает с python 3.11
try:
    from enum import StrEnum
except ImportError:
    from enum import Enum
    class StrEnum(str, Enum):
        pass
from typing import Iterable

from fire_uav.module_core.geometry import haversine_m
from fire_uav.utils.time import utc_now


class TargetTrackState(StrEnum):
    CANDIDATE = "CANDIDATE"
    CONFIRMED = "CONFIRMED"
    IN_ORBIT = "IN_ORBIT"
    ORBITED = "ORBITED"


@dataclass(slots=True)
class TargetObservation:
    class_label: str
    lat: float
    lon: float
    timestamp: datetime
    confidence: float = 0.0


@dataclass(slots=True)
class TargetTrack:
    track_id: int
    class_label: str
    lat: float
    lon: float
    last_seen_ts: datetime
    stable_count: int
    state: TargetTrackState = TargetTrackState.CANDIDATE
    is_stable: bool = False


@dataclass(slots=True)
class TrackUpdate:
    track: TargetTrack
    became_stable: bool
    should_confirm: bool


@dataclass(slots=True)
class _SuppressionZone:
    lat: float
    lon: float
    radius_m: float
    expires_at: datetime


class TargetTracker:
    def __init__(
        self,
        *,
        match_radius_m: float,
        suppression_radius_m: float,
        suppression_ttl_s: float,
        stable_frames_n: int,
        smooth_alpha: float = 0.35,
    ) -> None:
        self.match_radius_m = max(1.0, float(match_radius_m))
        self.suppression_radius_m = max(1.0, float(suppression_radius_m))
        self.suppression_ttl_s = max(1.0, float(suppression_ttl_s))
        self.stable_frames_n = max(1, int(stable_frames_n))
        self.smooth_alpha = max(0.01, min(1.0, float(smooth_alpha)))
        self.track_ttl_s = self.suppression_ttl_s

        self._next_track_id = 1
        self._tracks: dict[int, TargetTrack] = {}
        self._zones: list[_SuppressionZone] = []

    def reset(self) -> None:
        self._next_track_id = 1
        self._tracks.clear()
        self._zones.clear()

    def update(self, observations: Iterable[TargetObservation]) -> list[TrackUpdate]:
        obs_list = list(observations)
        if not obs_list:
            self._cleanup(utc_now())
            return []

        now = max(obs.timestamp for obs in obs_list)
        self._cleanup(now)
        out: list[TrackUpdate] = []

        for obs in obs_list:
            if self.is_suppressed(obs.lat, obs.lon, now=obs.timestamp):
                continue
            track = self._match_track(obs)
            if track is None:
                track = TargetTrack(
                    track_id=self._next_track_id,
                    class_label=obs.class_label,
                    lat=obs.lat,
                    lon=obs.lon,
                    last_seen_ts=obs.timestamp,
                    stable_count=1,
                )
                self._tracks[track.track_id] = track
                self._next_track_id += 1
            else:
                track.lat = self._smooth(track.lat, obs.lat)
                track.lon = self._smooth(track.lon, obs.lon)
                track.last_seen_ts = obs.timestamp
                track.stable_count += 1

            was_stable = track.is_stable
            if track.stable_count >= self.stable_frames_n:
                track.is_stable = True
                if track.state == TargetTrackState.CANDIDATE:
                    track.state = TargetTrackState.CONFIRMED

            should_confirm = (
                track.state == TargetTrackState.CONFIRMED and track.is_stable and not was_stable
            )
            out.append(
                TrackUpdate(
                    track=track,
                    became_stable=bool(track.is_stable and not was_stable),
                    should_confirm=bool(should_confirm),
                )
            )
        return out

    def mark_orbited(self, track_id: int, *, now: datetime | None = None) -> bool:
        ts = now or utc_now()
        self._cleanup(ts)
        track = self._tracks.get(int(track_id))
        if track is None:
            return False
        self.add_suppression_zone(lat=track.lat, lon=track.lon, now=ts)
        track.state = TargetTrackState.ORBITED
        return True

    def mark_in_orbit(self, track_id: int, *, now: datetime | None = None) -> bool:
        ts = now or utc_now()
        self._cleanup(ts)
        track = self._tracks.get(int(track_id))
        if track is None:
            return False
        self.add_suppression_zone(lat=track.lat, lon=track.lon, now=ts)
        track.state = TargetTrackState.IN_ORBIT
        return True

    def add_suppression_zone(self, *, lat: float, lon: float, now: datetime | None = None) -> None:
        ts = now or utc_now()
        self._cleanup(ts)
        self._zones.append(
            _SuppressionZone(
                lat=float(lat),
                lon=float(lon),
                radius_m=self.suppression_radius_m,
                expires_at=ts + timedelta(seconds=self.suppression_ttl_s),
            )
        )

    def is_suppressed(self, lat: float, lon: float, *, now: datetime | None = None) -> bool:
        ts = now or utc_now()
        self._cleanup(ts)
        return self._is_suppressed_no_cleanup(float(lat), float(lon))

    def _is_suppressed_no_cleanup(self, lat: float, lon: float) -> bool:
        for zone in self._zones:
            dist = haversine_m((zone.lat, zone.lon), (lat, lon))
            if dist <= zone.radius_m:
                return True
        return False

    def _match_track(self, obs: TargetObservation) -> TargetTrack | None:
        best: TargetTrack | None = None
        best_dist = float("inf")
        for track in self._tracks.values():
            if track.class_label != obs.class_label:
                continue
            age_s = (obs.timestamp - track.last_seen_ts).total_seconds()
            if age_s > self.track_ttl_s:
                continue
            dist = haversine_m((track.lat, track.lon), (obs.lat, obs.lon))
            if dist <= self.match_radius_m and dist < best_dist:
                best = track
                best_dist = dist
        return best

    def _cleanup(self, now: datetime) -> None:
        self._zones = [zone for zone in self._zones if zone.expires_at > now]

        stale_ids: list[int] = []
        for track_id, track in self._tracks.items():
            age_s = (now - track.last_seen_ts).total_seconds()
            if age_s > self.track_ttl_s:
                stale_ids.append(track_id)
                continue
            if track.state in (TargetTrackState.ORBITED, TargetTrackState.IN_ORBIT) and not self._is_suppressed_no_cleanup(
                track.lat, track.lon
            ):
                track.state = TargetTrackState.CONFIRMED
        for track_id in stale_ids:
            self._tracks.pop(track_id, None)

    def _smooth(self, prev: float, new: float) -> float:
        return (1.0 - self.smooth_alpha) * float(prev) + self.smooth_alpha * float(new)
