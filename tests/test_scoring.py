from wise.scoring.scoring import compute_case_scores
from wise.scoring.slices import aggregate_slices
from wise.norm import compute_view_weights


def test_case_scores_and_slices(simple_log, presence_norm):
    # one view
    view_name = presence_norm.get_view_names()[0]

    scores = compute_case_scores(
        df=simple_log,
        norm=presence_norm,
        view_name=view_name,
        case_id_col="case_id",
        activity_col="activity",
        timestamp_col="time",
    )

    # Case A has GR, case B does not
    a_score = float(scores.loc[scores["case_id"] == "A", "score"].iloc[0])
    b_score = float(scores.loc[scores["case_id"] == "B", "score"].iloc[0])
    assert a_score == 1.0
    assert b_score == 0.0

    # Use a simple slice: same for all cases, so PI should be 0
    slice_summary = aggregate_slices(
        df_scores=scores,
        df_log=simple_log,
        case_id_col="case_id",
        slice_cols=["activity"],  # not ideal slice, but enough for test
        shrink_k=0.0,
    )
    assert "PI" in slice_summary.columns
