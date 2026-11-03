import pathlib
import tarfile
import urllib.error
import urllib.request

import numpy as np
from numba import njit
from platformdirs import user_cache_dir

def find_track_dir(track_name: str) -> pathlib.Path:
    """
    Find the directory of the track map corresponding to the given track name.

    The repository ``maps/`` directory and the user cache directory are searched
    before downloading. A missing track is downloaded into the user cache.

    Args:
        track_name: name of the track (str).

    Returns:
        pathlib.Path: path to the track map directory.

    Raises:
        FileNotFoundError: if no map directory matching the track name is found.
    """
    map_dir = pathlib.Path(__file__).parent.parent.parent.parent / "maps"
    cache_map_dir = user_cache_dir("f1tenth_gym", False) / "maps"

    def _find_track_in(path):
        if path.is_dir():
            for subdir in path.iterdir():
                if track_name == subdir.stem.replace(" ", ""):
                    return subdir

    if not (track_dir := _find_track_in(map_dir) or _find_track_in(cache_map_dir)):
        print("Downloading Files for: " + track_name)
        tracks_url = "https://api.f1tenth.org/" + track_name + ".tar.xz"
        try:
            tracks_r = urllib.request.urlopen(tracks_url)
        except urllib.error.URLError as e:
            raise FileNotFoundError(f"No maps exists for {track_name}.") from e

        print("Extracting Files for: " + track_name)
        cache_map_dir.mkdir(parents=True, exist_ok=True)
        with tracks_r, tarfile.open(fileobj=tracks_r, mode="r|xz") as tracks_file:
            tracks_file.extractall(cache_map_dir, filter="data")
        track_dir = _find_track_in(cache_map_dir)

    if track_dir:
        return track_dir

    raise FileNotFoundError(f"no mapdir matching {track_name} in {[map_dir, cache_map_dir]}")


@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point: np.ndarray, trajectory: np.ndarray) -> tuple:
    """
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time
    constraints should not be an issue so long as trajectories are not insanely long.

        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

    Args:
        point: The 2d point (np.ndarray) to project onto the trajectory.
        trajectory: The trajectory (np.ndarray of shape (N, 2)) to project the point onto.
            The points must be unique. If they are not unique, a divide by 0 error will
            destroy the world.

    Returns:
        A 3-tuple ``(distance, t, min_dist_segment)``: the distance from the
        point to its projection on the trajectory, the fractional position of
        that projection within its segment (clipped to [0, 1]), and the index
        of the nearest segment.
    """
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1, :] + (t * diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))
    min_dist_segment = np.argmin(dists)
    # return (
    #     projections[min_dist_segment],
    #     dists[min_dist_segment],
    #     t[min_dist_segment],
    #     min_dist_segment,
    # )
    return (
        dists[min_dist_segment],
        t[min_dist_segment],
        min_dist_segment,
    )
