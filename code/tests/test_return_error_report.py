import math
import unittest

from training.report_return_errors import error_statistics, task_statistics


class ReturnErrorReportTests(unittest.TestCase):
    def test_percentage_point_units_population_variance_and_identities(self):
        m=error_statistics([0.,0.,0.],[0.,-.03,.02])
        self.assertAlmostEqual(m['bias_pp'],1/3)
        self.assertAlmostEqual(m['mae_pp'],5/3)
        self.assertAlmostEqual(m['error_variance_pp2'],38/9)
        self.assertAlmostEqual(m['absolute_error_variance_pp2'],14/9)
        self.assertAlmostEqual(m['mse_pp2'],m['error_variance_pp2']+m['bias_pp']**2)
        self.assertAlmostEqual(m['mse_pp2'],m['absolute_error_variance_pp2']+m['mae_pp']**2)
        self.assertAlmostEqual(m['error_std_pp']**2,m['error_variance_pp2'])

    def test_signed_errors_cancel_while_absolute_errors_do_not(self):
        m=error_statistics([.01,-.01],[0.,0.])
        self.assertEqual(m['bias_pp'],0.)
        self.assertEqual(m['mae_pp'],1.)
        self.assertEqual(m['error_variance_pp2'],1.)
        self.assertEqual(m['absolute_error_variance_pp2'],0.)

    def test_constant_predictions_shift_bias_but_do_not_reduce_error_variance(self):
        actual=[.01,-.02,.04]
        zero=error_statistics([0.]*3,actual)
        mean=error_statistics([.005]*3,actual)
        self.assertAlmostEqual(zero['error_variance_pp2'],mean['error_variance_pp2'])
        self.assertAlmostEqual(mean['bias_pp']-zero['bias_pp'],.5)
        single=error_statistics([0.],[.01])
        self.assertEqual(single['error_variance_pp2'],0.)

    def test_missing_or_nonfinite_pairs_are_rejected(self):
        for predicted,actual in [([],[]),([0.],[]),([math.nan],[0.]),([0.],[math.inf])]:
            with self.assertRaises(ValueError):error_statistics(predicted,actual)

    def test_zero_model_matches_zero_reference_exactly(self):
        rows=[{'expected_return':'0','actual_return':str(actual),'historical_mean_return':'.001',
               'date':f'2025-01-0{i+2}','label_end_date':f'2025-01-0{i+3}'}
              for i,actual in enumerate([.01,-.02,.03])]
        stats=task_statistics(rows)
        self.assertEqual(stats['selected'],stats['zero'])
        self.assertEqual(stats['mae_improvement_vs_zero_pp'],0.)
        self.assertEqual(stats['variance_reduction_vs_zero_pp2'],0.)


if __name__=='__main__':unittest.main()
