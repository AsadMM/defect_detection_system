def group_test_results(test_results: list[tuple[float, str]]) -> dict[str, dict[str, int]]:
    """
    Group per-image test results by defect class.

    For defects the correct class is 'Predicted_1'.
    For good the correct class is 'Predicted_0'.
    """
    test_results_grouped = {}
    for res, defect in test_results:
        if test_results_grouped.get(defect) is None:
            test_results_grouped[defect] = {'Total': 0, 'Predicted_0': 0, 'Predicted_1': 0}
        test_results_grouped[defect]['Total'] += 1
        if res == 0:
            test_results_grouped[defect]['Predicted_0'] += 1
        else:
            test_results_grouped[defect]['Predicted_1'] += 1
    return test_results_grouped
