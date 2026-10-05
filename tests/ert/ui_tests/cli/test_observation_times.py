"""Tests behavior of matching response times to observation times"""

from contextlib import redirect_stderr
from datetime import date, datetime, timedelta
from io import StringIO
from textwrap import dedent

import hypothesis.strategies as st
import numpy as np
import polars as pl
import pytest
from hypothesis import assume, given, settings
from resfo_utilities.testing import (
    Date,
    Simulator,
    Smspec,
    SmspecIntehead,
    SummaryMiniStep,
    SummaryStep,
    UnitSystem,
    Unsmry,
    summaries,
)

from ert.cli.main import ErtCliError
from ert.config import BreakthroughConfig, SummaryConfig
from ert.mode_definitions import ENSEMBLE_SMOOTHER_MODE, ES_MDA_MODE
from ert.storage import open_storage
from tests.ert.unit_tests.config.observations_generator import (
    as_obs_config_content,
    summary_observations,
)

from .run_cli import run_cli

start = datetime(1969, 1, 1)  # ruff: ignore[call-datetime-without-tzinfo]
observation_times = st.dates(
    min_value=date.fromordinal((start + timedelta(hours=1)).toordinal()),
    max_value=date(2024, 1, 1),
).map(lambda x: datetime.fromordinal(x.toordinal()))


@pytest.mark.filterwarnings(
    "ignore:.*overflow encountered in multiply.*:RuntimeWarning"
)
@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
@settings(max_examples=3)
@given(
    responses_observation=observation_times.flatmap(
        lambda observation_time: st.fixed_dictionaries(
            {
                "responses": st.lists(
                    summaries(
                        start_date=st.just(start),
                        time_deltas=st.just(
                            [((observation_time - start).total_seconds()) / 3600]
                        ),
                        summary_keys=st.just(["FOPR"]),
                        use_days=st.just(False),
                    ),
                    min_size=2,
                    max_size=2,
                ),
                "observation": summary_observations(
                    summary_keys=st.just("FOPR"),
                    std_cutoff=10.0,
                    names=st.just("FOPR_OBSERVATION"),
                    datetimes=st.just(observation_time),
                ),
            }
        )
    ),
    std_cutoff=st.floats(min_value=1e-6, max_value=1.0),
    enkf_alpha=st.floats(min_value=3.0, max_value=10.0),
    epsilon=st.sampled_from([0.0, 1.1, 2.0, -2.0]),
)
def test_that_small_time_mismatches_in_summaries_are_ignored(
    responses_observation, tmp_path_factory, std_cutoff, enkf_alpha, epsilon
):
    responses = responses_observation["responses"]
    observation = responses_observation["observation"]
    tmp_path = tmp_path_factory.mktemp("summary")
    (tmp_path / "config.ert").write_text(
        dedent(
            f"""
            NUM_REALIZATIONS 2
            QUEUE_SYSTEM LOCAL
            QUEUE_OPTION LOCAL MAX_RUNNING 2
            ECLBASE CASE
            SUMMARY FOPR
            MAX_SUBMIT 1
            GEN_KW KW_NAME prior.txt
            OBS_CONFIG observations.txt
            STD_CUTOFF {std_cutoff}
            ENKF_ALPHA {enkf_alpha}
            """
        )
    )

    # Add some inprecision to the reported time
    for r in responses:
        r[1].steps[-1].ministeps[-1].params[0] += epsilon

    (tmp_path / "prior.txt").write_text("KW_NAME NORMAL 0 1")
    response_values = np.array(
        [r[1].steps[-1].ministeps[-1].params[-1] for r in responses]
    )
    std_dev = response_values.std(ddof=0)
    assume(np.isfinite(std_dev))
    assume(std_dev > std_cutoff)
    observation.value = float(response_values.mean())
    for i in range(2):
        for j in range(4):
            summary = responses[i]
            smspec, unsmry = summary
            (tmp_path / f"simulations/realization-{i}/iter-{j}").mkdir(parents=True)
            smspec.to_file(
                tmp_path / f"simulations/realization-{i}/iter-{j}/CASE.SMSPEC"
            )
            unsmry.to_file(
                tmp_path / f"simulations/realization-{i}/iter-{j}/CASE.UNSMRY"
            )
    (tmp_path / "observations.txt").write_text(as_obs_config_content(observation))

    if abs(epsilon) < 1 / 3600:  # less than one second
        stderr = StringIO()
        with redirect_stderr(stderr):
            run_cli(
                ES_MDA_MODE,
                str(tmp_path / "config.ert"),
                "--weights=2,1",
            )
        assert "Experiment completed" in stderr.getvalue()
    else:
        with pytest.raises(ErtCliError, match="No active observations"):
            run_cli(
                ES_MDA_MODE,
                "--disable-monitoring",
                str(tmp_path / "config.ert"),
                "--weights=2,1",
            )


def _write_summary_files(
    directory,
    *,
    times_days: list[float],
    fopr_values: list[float],
    wwct_values: list[float],
    start_date: datetime,
):
    smspec = Smspec(
        intehead=SmspecIntehead(
            unit=UnitSystem.METRIC, simulator=Simulator.ECLIPSE_100
        ),
        restart="        ",
        num_keywords=3,
        nx=1,
        ny=1,
        nz=1,
        restarted_from_step=0,
        keywords=["TIME    ", "FOPR", "WWCT"],
        well_names=[":+:+:+:+", "        ", "OP1"],
        region_numbers=[-32676, 0, 0],
        units=["DAYS    ", "SM3/DAY ", "SM3/SM3 "],
        start_date=Date.from_datetime(start_date),
    )
    unsmry = Unsmry(
        steps=[
            SummaryStep(
                seqnum=i,
                ministeps=[
                    SummaryMiniStep(
                        mini_step=i,
                        params=[time_days, fopr, wwct],
                    )
                ],
            )
            for i, (time_days, fopr, wwct) in enumerate(
                zip(times_days, fopr_values, wwct_values, strict=True)
            )
        ]
    )
    directory.mkdir(parents=True, exist_ok=True)
    smspec.to_file(directory / "CASE.SMSPEC")
    unsmry.to_file(directory / "CASE.UNSMRY")


@pytest.mark.filterwarnings(
    "ignore:Config contains a SUMMARY key but no forward model.*:UserWarning"
)
@pytest.mark.parametrize(
    ("threshold", "expected_breakthrough_days"),
    [
        pytest.param(0.05, {0: 30, 1: 30}, id="both realizations reach threshold"),
        pytest.param(0.2, {0: 120, 1: 151}, id="one realization reaches threshold"),
        pytest.param(0.9, {0: 151, 1: 151}, id="neither realization reaches threshold"),
    ],
)
def test_that_realizations_with_different_end_dates_sets_observation_values(
    tmp_path, threshold, expected_breakthrough_days
):
    """Two realizations simulate for different lengths of time.

    There are three SUMMARY_OBSERVATIONs, one at a date where
    both realizations have simulated data, one at a date only the longer
    realization has reached, and one at a date neither realization has
    reached; and a single BREAKTHROUGH_OBSERVATION whose threshold is
    reached by both, one, or neither realization based on the threshold parameter.
    """
    start_date = datetime(2000, 1, 1)  # ruff: ignore[call-datetime-without-tzinfo]

    # Realization 0 is simulated for 5 months.
    times_days0 = [30, 60, 90, 120, 150]
    fopr_values0 = [300, 280, 260, 240, 220]
    realization0_kwargs = {
        "times_days": times_days0,
        "fopr_values": fopr_values0,
        "wwct_values": [0.05, 0.10, 0.15, 0.25, 0.35],
        "start_date": start_date,
    }
    # Realization 1 is only simulated for 3 months, i.e. it has an earlier
    # end date than realization 0.
    times_days1 = [30, 60, 90]
    fopr_values1 = [305, 285, 265]
    realization1_kwargs = {
        "times_days": times_days1,
        "fopr_values": fopr_values1,
        "wwct_values": [0.05, 0.10, 0.15],
        "start_date": start_date,
    }

    # Both realizations have data at day 60.
    both_obs_date = start_date + timedelta(days=60)
    # Only realization 0 (the later end date) has data at day 120.
    one_obs_date = start_date + timedelta(days=120)
    # Neither realization has data at day 200 (beyond both end dates).
    neither_obs_date = start_date + timedelta(days=200)
    breakthrough_obs_date = start_date + timedelta(days=100)

    (tmp_path / "config.ert").write_text(
        dedent(
            """
            NUM_REALIZATIONS 2
            QUEUE_SYSTEM LOCAL
            QUEUE_OPTION LOCAL MAX_RUNNING 2
            ECLBASE CASE
            MAX_SUBMIT 1
            GEN_KW KW_NAME prior.txt
            OBS_CONFIG observations.txt
            STD_CUTOFF 0.0001
            ENKF_ALPHA 5.0
            """
        )
    )
    (tmp_path / "prior.txt").write_text("KW_NAME NORMAL 0 1")
    (tmp_path / "observations.txt").write_text(
        dedent(
            f"""
            SUMMARY_OBSERVATION FOPR_BOTH_OBS {{
                KEY = FOPR;
                VALUE = 282.5;
                ERROR = 5;
                DATE = {both_obs_date:%Y-%m-%d};
            }};
            SUMMARY_OBSERVATION FOPR_ONE_OBS {{
                KEY = FOPR;
                VALUE = 240;
                ERROR = 5;
                DATE = {one_obs_date:%Y-%m-%d};
            }};
            SUMMARY_OBSERVATION FOPR_NEITHER_OBS {{
                KEY = FOPR;
                VALUE = 200;
                ERROR = 5;
                DATE = {neither_obs_date:%Y-%m-%d};
            }};
            BREAKTHROUGH_OBSERVATION BRT_OBS {{
                KEY=WWCT:OP1;
                DATE={breakthrough_obs_date:%Y-%m-%d};
                ERROR=30; -- days
                THRESHOLD={threshold};
            }};
            """
        )
    )

    for iteration in (0, 1):
        _write_summary_files(
            tmp_path / f"simulations/realization-0/iter-{iteration}",
            **realization0_kwargs,
        )
        _write_summary_files(
            tmp_path / f"simulations/realization-1/iter-{iteration}",
            **realization1_kwargs,
        )

    stderr = StringIO()
    with redirect_stderr(stderr):
        run_cli(
            ENSEMBLE_SMOOTHER_MODE,
            str(tmp_path / "config.ert"),
        )
    assert "Experiment completed" in stderr.getvalue()

    with open_storage(tmp_path / "storage") as storage:
        experiment = next(iter(storage.experiments))
        prior = experiment.get_ensemble_by_name("iter-0")

        # Verify breakthrough observations
        breakthrough_responses = prior.load_responses("breakthrough", (0, 1))
        values_by_realization = dict(
            zip(
                breakthrough_responses["realization"].to_list(),
                breakthrough_responses["values"].to_list(),
                strict=True,
            )
        )
        for realization, breakthrough_day in expected_breakthrough_days.items():
            breakthrough_date = start_date + timedelta(days=breakthrough_day)
            assert values_by_realization[realization] == pytest.approx(
                (breakthrough_date - breakthrough_obs_date).days
            )

        # Verify summary observations
        obs_and_responses = prior.get_observations_and_responses(
            ["FOPR_BOTH_OBS", "FOPR_ONE_OBS", "FOPR_NEITHER_OBS"],
            np.array([0, 1]),
        )
        responses_by_obs = {
            row["observation_key"]: (row["0"], row["1"])
            for row in obs_and_responses.to_dicts()
        }
        # Both realizations have simulated data at day 60.
        real0, real1 = responses_by_obs["FOPR_BOTH_OBS"]
        assert real0 == pytest.approx(fopr_values0[times_days0.index(60)])
        assert real1 == pytest.approx(fopr_values1[times_days1.index(60)])
        # Only realization 0 has simulated data at day 120.
        real0, real1 = responses_by_obs["FOPR_ONE_OBS"]
        assert real0 == pytest.approx(fopr_values0[times_days0.index(120)])
        assert real1 is None
        # Neither realization has simulated data at day 200.
        real0, real1 = responses_by_obs["FOPR_NEITHER_OBS"]
        assert real0 is None
        assert real1 is None


def test_that_breakthrough_values_are_recomputed_as_more_realizations_complete(
    tmp_path,
):
    """Forward models finish at different times, so the ensemble end date
    (used to fill in breakthrough values for realizations whose threshold
    was never reached) is not knowable until all realizations with
    responses have been inspected.

    In this test, realization 1 (the shorter simulation) finishes first: its
    breakthrough value is filled using the ensemble end date *at the time it is
    loaded*, and recomputed  once realization 0 (the longer simulation)
    completes.
    """
    start_date = datetime(2000, 1, 1)  # ruff: ignore[call-datetime-without-tzinfo]
    threshold = 0.9  # never reached by either realization
    breakthrough_obs_date = start_date + timedelta(days=100)

    # Realization 0 finishes later, after 5 months.
    times_days0 = [30, 60, 90, 120, 150]
    wwct_values0 = [0.05, 0.10, 0.15, 0.25, 0.35]
    # Realization 1 finishes first, after 3 months.
    times_days1 = [30, 60, 90]
    wwct_values1 = [0.05, 0.10, 0.15]

    breakthrough_config = BreakthroughConfig(
        keys=["BREAKTHROUGH:WWCT:OP1"],
        summary_keys=["WWCT:OP1"],
        thresholds=[threshold],
        observed_dates=[breakthrough_obs_date],
    )

    def summary_response(realization: int, times_days: list[int], wwct_values):
        return pl.DataFrame(
            {
                "realization": [realization] * len(times_days),
                "response_key": ["WWCT:OP1"] * len(times_days),
                "time": [start_date + timedelta(days=d) for d in times_days],
                "values": wwct_values,
            }
        )

    with open_storage(tmp_path / "storage", mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "response_configuration": [
                    SummaryConfig(keys=["*"], input_files=["not_relevant"]).model_dump(
                        mode="json"
                    ),
                    breakthrough_config.model_dump(mode="json"),
                ],
            }
        )
        ensemble = storage.create_ensemble(
            experiment, ensemble_size=2, iteration=0, name="prior"
        )

        # Realization 1 finishes first.
        ensemble.save_response(
            "summary", summary_response(1, times_days1, wwct_values1), 1
        )
        ensemble.refresh_ensemble_state()
        breakthrough_response_1 = breakthrough_config.derive_from_storage(
            0, 1, ensemble
        )
        ensemble.save_response("breakthrough", breakthrough_response_1, 1)

        early_breakthrough_value = ensemble.load_responses("breakthrough", (1,))[
            "values"
        ].item()
        early_expected_offset = (
            start_date + timedelta(days=times_days1[-1] + 1) - breakthrough_obs_date
        ).days
        assert early_breakthrough_value == pytest.approx(early_expected_offset)

        # Realization 0 now completes, extending the ensemble end date.
        ensemble.save_response(
            "summary", summary_response(0, times_days0, wwct_values0), 0
        )
        ensemble.refresh_ensemble_state()
        breakthrough_response_0 = breakthrough_config.derive_from_storage(
            0, 0, ensemble
        )
        ensemble.save_response("breakthrough", breakthrough_response_0, 0)

        # Reloading both realizations' (already saved) breakthrough responses
        # must reflect the *new* ensemble end date
        updated_breakthrough = ensemble.load_responses("breakthrough", (0, 1))
        updated_breakthrough_values = dict(
            zip(
                updated_breakthrough["realization"].to_list(),
                updated_breakthrough["values"].to_list(),
                strict=True,
            )
        )
        updated_expected_offset = (
            start_date + timedelta(days=times_days0[-1] + 1) - breakthrough_obs_date
        ).days
        assert updated_breakthrough_values[0] == pytest.approx(updated_expected_offset)
        assert updated_breakthrough_values[1] == pytest.approx(updated_expected_offset)
        assert updated_breakthrough_values[1] != pytest.approx(early_breakthrough_value)
