import numpy as np
import tomsup as ts
from tomsup.ktom_functions import (
    learning_function,
    expected_payoff_fun,
    p_op0_fun,
    param_var_update,
    p_k_udpate,
    p_opk_approx_fun,
    p_op_var0_update,
)


def test_learning_function():
    penny = ts.PayoffMatrix(name="penny_competitive")
    prev_internal_states = {
        "opponent_states": {},
        "own_states": {"p_op_mean0": 0, "p_op_var0": 0},
    }
    params = {"volatility": -2, "b_temp": -1}
    outcome = learning_function(
        prev_internal_states,
        params,
        self_choice=1,
        op_choice=1,
        level=0,
        agent=0,
        p_matrix=penny,
    )
    assert abs(outcome["own_states"]["p_op_mean0"] - 0.44216598162254866) < 0.01
    assert abs(outcome["own_states"]["p_op_var0"] - -0.12292276280308079) < 0.01


def test_expected_payoff_fun():
    staghunt = ts.PayoffMatrix(name="staghunt")
    assert expected_payoff_fun(1, agent=0, p_matrix=staghunt) == 2


def test_p_op0_fun():
    assert abs(p_op0_fun(p_op_mean0=0.7, p_op_var0=0.3) - 0.6397417553178626) < 0.01


def test_p_k_udpate():
    arr = p_k_udpate(
        prev_p_k=np.array([1.0]),
        p_opk_approx=np.array([-0.69314718]),
        op_choice=1,
        dilution=None,
    )
    assert arr == np.array([1.0])


def test_p_opk_approx_fun():
    arr = p_opk_approx_fun(
        prev_p_op_mean=np.array([0]),
        prev_param_var=np.array([[0, 0, 0]]),
        prev_gradient=np.array([[0, 0, 0]]),
        level=1,
    )
    assert abs(arr - np.array([-0.69314718])) < 0.01
