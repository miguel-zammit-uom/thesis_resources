import datetime


def _information_print(n_gen, n_pareto, avg_clf_metric, max_clf_metric, avg_imp_score, max_imp_score, avg_clf_acc,
                       max_clf_acc, tau_imp, tau_clf, output_txt_filename, display=False):
    file = open(output_txt_filename, "a")

    if n_gen == 1:
        if display:
            print("===================================================================================================="
                  "============================")
            print("    n_gen    ||   n_pareto   |   avg_Jup_acc    |   max_Jup_acc   | avg_imp_score | max_imp_score  |"
                  " tau_impute  |    tau_clf    |")
            print("==================================================================================================="
                  "====")

        file.write(f'\nGenetic Algorithm Run Initiated on {datetime.datetime.now()}\n')
        file.write("=================================================================================================="
                   "======================================================================")
        file.write("\n    n_gen    ||   n_pareto   |  avg_clf_metric  |  max_clf_metric  | avg_imp_score | "
                   "max_imp_score  |   avg_Jup_acc    |   max_Jup_acc    |  tau_impute  |    tau_clf    |")
        file.write("\n================================================================================================="
                   "=======================================================================")
        if display:
            print(
                f"     {n_gen:03d}     ||      {n_pareto:03d}     |     {avg_clf_metric:+.4f}      |     {max_clf_metric:+.4f}   "
                f"   |     {avg_imp_score:+.4f}   |     {max_imp_score:+.4f}    |     {avg_clf_acc:+.4f}      |     "
                f"{max_clf_acc:+.4f}      |     ----     |    ----  ")
        file.write(
            f"\n     {n_gen:03d}     ||      {n_pareto:03d}     |     {avg_clf_metric:+.4f}      |     {max_clf_metric:+.4f}   "
                f"   |     {avg_imp_score:+.4f}   |     {max_imp_score:+.4f}    |     {avg_clf_acc:+.4f}      |     "
                f"{max_clf_acc:+.4f}      |     ----     |     ----    "
            f"  |")
    else:
        if display:
            print(f"     {n_gen:03d}     ||      {n_pareto:03d}     |     {avg_clf_metric:+.4f}     |     {max_clf_metric:+.4f}   "
                  f"  |     {avg_imp_score:+.4f}    |     {max_imp_score:+.4f}    |     {avg_clf_acc:+.4f}      |     "
                f"{max_clf_acc:+.4f}      |")
        file.write(
            f"\n     {n_gen:03d}     ||      {n_pareto:03d}     |     {avg_clf_metric:+.4f}      |     {max_clf_metric:+.4f}   "
            f"   |     {avg_imp_score:+.4f}   |     {max_imp_score:+.4f}    |     {avg_clf_acc:+.4f}      |     "
                f"{max_clf_acc:+.4f}      |   {tau_imp:+.4f}    |   {tau_clf:+.4f}   "
                   f"  |")


def _upr_border_print():
    print('==============================================================================')


def _lwr_border_print():
    print('==============================================================================\n\n')


def _mid_border_print():
    print('------------------------------------------------------------------------------')
