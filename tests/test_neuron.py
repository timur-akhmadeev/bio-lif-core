"""
Тесты для LIFNeuron.

Проверяем:
    1. Физику базовой LIF-модели.
    2. Точное аналитическое интегрирование
       (полугрупповое свойство, независимость от dt).
    3. Пороговый ток и поведение на границе
       (в том числе при крупных dt — без ложных спайков
       из-за округления float).
    4. Поведение ниже / на / выше порога.
    5. Семантику spike/reset/refractory.
    6. Пересечение границы рефрактерности внутри dt.
    7. Время спайков и ISI относительно аналитики:
       отклонение ограничено одним шагом dt.
    8. Сходимость спайк-тайминга по dt.
    9. Защиту от NaN/inf/переполнения/некорректных параметров;
       ошибка валидации не меняет состояние нейрона.
   10. Utility API:
       threshold_current, steady_state,
       is_refractory, reset, repr.
"""

import numpy as np
import pytest

from biomorph import LIFNeuron


# =====================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =====================================================================


def _simulate_constant_current(
    dt: float,
    duration: float,
    current: float,
    **neuron_kwargs,
):
    """
    Возвращает времена зарегистрированных спайков.

    Время spike определяется дискретной семантикой
    step(): t + dt.

    Время считается как i * dt, а не накоплением t += dt:
    так на длинных прогонах не копится ошибка округления
    и число шагов не «плавает» на единицу.
    """
    n = LIFNeuron(**neuron_kwargs)

    n_steps = int(round(duration / dt))
    spike_times = []

    for i in range(n_steps):
        t = i * dt

        if n.step(
            input_current=current,
            t=t,
            dt=dt,
        ):
            spike_times.append(t + dt)

    return np.asarray(spike_times)


def _analytic_first_crossing(
    n: LIFNeuron,
    current: float,
) -> float:
    """
    Время (мс) достижения порога из v_rest при постоянном I > I_th:

        t = tau_m * ln((V_inf - V_rest) / (V_inf - V_threshold))
    """
    v_inf = n.steady_state(current)

    return n.tau_m * np.log(
        (v_inf - n.v_rest)
        /
        (v_inf - n.v_threshold)
    )


def _analytic_isi(
    n: LIFNeuron,
    current: float,
) -> float:
    """
    Межспайковый интервал (мс) при постоянном I > I_th:

        ISI = tau_m * ln((V_inf - V_reset) / (V_inf - V_threshold))
              + refractory
    """
    v_inf = n.steady_state(current)

    return (
        n.tau_m
        * np.log(
            (v_inf - n.v_reset)
            /
            (v_inf - n.v_threshold)
        )
        + n.refractory
    )


# =====================================================================
# БАЗОВЫЕ СЦЕНАРИИ
# =====================================================================


def test_rest_without_current():
    """Без входного тока потенциал остаётся на v_rest, спайков нет."""
    n = LIFNeuron()

    for step in range(1000):
        t = step * 0.1

        spike = n.step(
            input_current=0.0,
            t=t,
            dt=0.1,
        )

        assert spike is False

    assert n.v == pytest.approx(
        n.v_rest,
        abs=1e-12,
    )


def test_steady_state_is_reached_below_threshold():
    """
    При I < I_th:

        V -> V_inf < V_threshold

    Спайков нет.

    После 500 мс остаточный экспоненциальный член
    уже очень мал, поэтому используем допуск 1e-8 мВ.
    """
    n = LIFNeuron()
    current = 0.9 * n.threshold_current

    for step in range(5000):
        t = step * 0.1

        assert n.step(
            input_current=current,
            t=t,
            dt=0.1,
        ) is False

    v_inf = n.steady_state(current)

    assert v_inf < n.v_threshold

    assert n.v == pytest.approx(
        v_inf,
        abs=1e-8,
    )


@pytest.mark.parametrize(
    "dt",
    [0.1, 1.0, 20.0, 1000.0],
)
def test_exact_threshold_current_does_not_spike_in_finite_time(dt):
    """
    При I == I_th:

        V_inf = V_threshold

    Но потенциал достигает порога только асимптотически.
    Поэтому за конечное время спайка нет — при любом dt.

    Крупные dt здесь принципиальны. При dt порядка tau_m и больше
    поправка (V - V_inf) * exp(-dt / tau_m) становится меньше
    половины ulp, и float-сумма округляется ровно до V_inf ==
    V_threshold — то есть ложный спайк. step() обязан этого
    не допускать.
    """
    n = LIFNeuron()
    current = n.threshold_current

    assert n.steady_state(current) == pytest.approx(
        n.v_threshold
    )

    total_time = 2000.0  # = 100 * tau_m
    n_steps = max(2, int(round(total_time / dt)))

    for i in range(n_steps):
        assert n.step(
            input_current=current,
            t=i * dt,
            dt=dt,
        ) is False

    assert n.v < n.v_threshold


def test_above_threshold_current_generates_spikes():
    """При I > I_th нейрон генерирует регулярные спайки."""
    n = LIFNeuron()
    current = 1.5 * n.threshold_current

    spike_count = 0

    for step in range(2000):
        t = step * 0.1

        if n.step(
            input_current=current,
            t=t,
            dt=0.1,
        ):
            spike_count += 1

    assert spike_count > 5


# =====================================================================
# ТОЧНОЕ АНАЛИТИЧЕСКОЕ ИНТЕГРИРОВАНИЕ
# =====================================================================


def test_exact_subthreshold_update():
    """
    Проверяем формулу exact integration:

        V_inf = V_rest + R_m * I

        V(dt) =
            V_inf +
            (V0 - V_inf) * exp(-dt / tau_m)
    """
    n = LIFNeuron()

    input_current = 0.1
    dt = 5.0

    v0 = n.v
    v_inf = n.steady_state(input_current)

    expected = (
        v_inf
        + (v0 - v_inf)
        * np.exp(-dt / n.tau_m)
    )

    spike = n.step(
        input_current=input_current,
        t=0.0,
        dt=dt,
    )

    assert spike is False

    assert n.v == pytest.approx(
        expected,
        rel=1e-13,
        abs=1e-13,
    )


def test_exact_integration_composition():
    """
    Полугрупповое свойство точного интегратора:

        step(dt) + step(dt) == step(2 * dt)

    При постоянном токе и отсутствии spike/reset.
    """
    input_current = 0.1
    dt = 2.0

    n1 = LIFNeuron()
    n2 = LIFNeuron()

    n1.step(
        input_current=input_current,
        t=0.0,
        dt=dt,
    )

    n1.step(
        input_current=input_current,
        t=dt,
        dt=dt,
    )

    n2.step(
        input_current=input_current,
        t=0.0,
        dt=2.0 * dt,
    )

    assert n1.v == pytest.approx(
        n2.v,
        rel=1e-13,
        abs=1e-13,
    )


@pytest.mark.parametrize(
    "dt",
    [0.01, 0.1, 1.0, 5.0, 25.0],
)
def test_subthreshold_result_independent_of_dt(dt):
    """
    Главное свойство точного интегратора: при постоянном токе
    ниже порога V(T) не зависит от dt.

    (Явный Эйлер дал бы заметную зависимость от dt; здесь
    остаётся только ошибка округления float.)

    T = 100 мс кратно всем dt из списка.
    """
    n = LIFNeuron()
    current = 0.5 * n.threshold_current

    total_time = 100.0
    n_steps = int(round(total_time / dt))

    for i in range(n_steps):
        assert n.step(
            input_current=current,
            t=i * dt,
            dt=dt,
        ) is False

    v_inf = n.steady_state(current)

    expected = (
        v_inf
        + (n.v_rest - v_inf)
        * np.exp(-total_time / n.tau_m)
    )

    assert n.v == pytest.approx(
        expected,
        abs=1e-9,
    )


# =====================================================================
# ПОРОГ
# =====================================================================


def test_spike_when_initial_v_equals_threshold():
    """Если v уже равен порогу, условие >= даёт спайк."""
    n = LIFNeuron()

    n.v = n.v_threshold

    spike = n.step(
        input_current=0.0,
        t=0.0,
        dt=0.1,
    )

    assert spike is True
    assert n.v == pytest.approx(n.v_reset)


def test_threshold_current_value():
    """threshold_current = (V_th - V_rest) / R_m."""
    n = LIFNeuron(
        tau_m=20.0,
        r_m=100.0,
        v_rest=-70.0,
        v_threshold=-55.0,
    )

    expected = (
        n.v_threshold - n.v_rest
    ) / n.r_m

    assert n.threshold_current == pytest.approx(
        expected
    )

    assert n.threshold_current == pytest.approx(
        0.15
    )


def test_steady_state_formula():
    """steady_state = V_rest + R_m * I."""
    n = LIFNeuron(
        r_m=100.0,
        v_rest=-70.0,
    )

    assert n.steady_state(0.2) == pytest.approx(
        -50.0
    )

    assert n.steady_state(0.0) == pytest.approx(
        -70.0
    )


def test_spike_from_integration_sets_reset_and_refractory():
    """
    Порог пересекается внутри шага (post-check), а не стоит
    на нём с самого начала (pre-check).

    Спайк регистрируется на t + dt, рефрактерность
    отсчитывается от этого момента.
    """
    n = LIFNeuron(
        refractory=5.0
    )

    current = 2.0 * n.threshold_current

    # Чуть ниже порога: без интегрирования спайка бы не было.
    n.v = n.v_threshold - 0.1

    spike = n.step(
        input_current=current,
        t=10.0,
        dt=1.0,
    )

    assert spike is True

    assert n.v == pytest.approx(
        n.v_reset
    )

    assert n.refractory_until == pytest.approx(
        10.0 + 1.0 + 5.0
    )


def test_large_step_does_not_miss_threshold_crossing():
    """
    Порог пересекается в середине большого шага.

    При постоянном токе V(t) монотонна, поэтому проверка
    в конце шага не может пропустить пересечение внутри шага.
    Спайк регистрируется на t + dt (дискретная семантика).
    """
    n = LIFNeuron()
    current = 1.5 * n.threshold_current

    # Из v_rest порог достигается примерно за 22 мс,
    # а шаг покрывает [0, 100].
    assert _analytic_first_crossing(n, current) < 100.0

    spike = n.step(
        input_current=current,
        t=0.0,
        dt=100.0,
    )

    assert spike is True

    assert n.v == pytest.approx(
        n.v_reset
    )

    assert n.refractory_until == pytest.approx(
        100.0 + n.refractory
    )


# =====================================================================
# SPIKE / RESET / СЕМАНТИКА ВРЕМЕНИ
# =====================================================================


def test_spike_resets_voltage():
    """После спайка потенциал ровно v_reset."""
    n = LIFNeuron()

    n.v = n.v_threshold

    spike = n.step(
        input_current=0.0,
        t=10.0,
        dt=0.1,
    )

    assert spike is True

    assert n.v == pytest.approx(
        n.v_reset
    )


def test_refractory_starts_after_spike_time():
    """
    Спайк фиксируется в конце шага (t + dt),
    и рефрактерность отсчитывается от этого момента:

        refractory_until = t + dt + refractory
    """
    n = LIFNeuron(
        refractory=5.0
    )

    n.v = n.v_threshold

    spike = n.step(
        input_current=0.0,
        t=10.0,
        dt=0.5,
    )

    assert spike is True

    assert n.refractory_until == pytest.approx(
        10.0 + 0.5 + 5.0
    )


@pytest.mark.parametrize(
    "dt",
    [0.01, 0.1, 0.5, 1.0, 5.0],
)
def test_first_spike_time_within_one_dt_of_analytic_crossing(dt):
    """
    Из v_rest при постоянном I > I_th:

        t_cross = tau_m * ln((V_inf - V_rest) / (V_inf - V_threshold))

    Спайк регистрируется на правой границе шага и не раньше
    истинного пересечения, поэтому

        t_cross <= t_spike < t_cross + dt

    Это проверка и точности интегратора, и семантики времени.
    """
    n = LIFNeuron()
    current = 1.5 * n.threshold_current

    t_cross = _analytic_first_crossing(n, current)

    spikes = _simulate_constant_current(
        dt=dt,
        duration=50.0,
        current=current,
    )

    assert len(spikes) >= 1

    assert t_cross - 1e-9 <= spikes[0] <= t_cross + dt + 1e-9


# =====================================================================
# РЕФРАКТЕРНОСТЬ
# =====================================================================


def test_refractory_blocks_spikes():
    """Во время абсолютной рефрактерности спайков нет."""
    n = LIFNeuron(
        refractory=5.0
    )

    current = 2.0 * n.threshold_current

    n.v = n.v_threshold

    assert n.step(
        input_current=current,
        t=0.0,
        dt=0.1,
    ) is True

    assert n.refractory_until == pytest.approx(
        5.1
    )

    for step in range(1, 50):
        t = step * 0.1

        spike = n.step(
            input_current=current,
            t=t,
            dt=0.1,
        )

        assert spike is False

        assert n.v == pytest.approx(
            n.v_reset
        )


def test_is_refractory_boundaries():
    """Проверка точного поведения на границе refractory_until."""
    n = LIFNeuron()

    n.refractory_until = 10.0

    assert n.is_refractory(9.999999) is True
    assert n.is_refractory(10.0) is False
    assert n.is_refractory(10.000001) is False


def test_refractory_spanning_step_exact_value():
    """
    Шаг [2, 4] ms при refractory_until = 3 ms:

        2 -> 3 ms:
            нейрон заморожен на v_reset

        3 -> 4 ms:
            точное интегрирование на 1 ms
    """
    n = LIFNeuron(
        tau_m=20.0,
        r_m=100.0,
        v_rest=-70.0,
        v_threshold=-55.0,
        v_reset=-75.0,
        refractory=5.0,
    )

    n.refractory_until = 3.0
    n.v = n.v_reset

    current = 0.5 * n.threshold_current
    v_inf = n.steady_state(current)

    expected = (
        v_inf
        + (n.v_reset - v_inf)
        * np.exp(-1.0 / n.tau_m)
    )

    spike = n.step(
        input_current=current,
        t=2.0,
        dt=2.0,
    )

    assert spike is False

    assert n.v == pytest.approx(
        expected,
        rel=1e-13,
        abs=1e-13,
    )

    assert n.v > n.v_reset


def test_refractory_entire_step_keeps_reset():
    """Если весь шаг внутри refractory — мембрана не интегрируется."""
    n = LIFNeuron()

    n.refractory_until = 10.0

    # Намеренно отличается от reset.
    n.v = -60.0

    spike = n.step(
        input_current=10.0,
        t=8.0,
        dt=1.0,
    )

    assert spike is False

    assert n.v == pytest.approx(
        n.v_reset
    )


def test_step_ending_exactly_at_refractory_end_stays_frozen():
    """
    Шаг [9, 10] при refractory_until = 10:

        t_end == refractory_until
        -> весь шаг ещё внутри рефрактерности.
    """
    n = LIFNeuron()

    n.refractory_until = 10.0

    # Намеренно отличается от reset.
    n.v = -60.0

    spike = n.step(
        input_current=10.0,
        t=9.0,
        dt=1.0,
    )

    assert spike is False

    assert n.v == pytest.approx(
        n.v_reset
    )


def test_step_starting_exactly_at_refractory_end_integrates_fully():
    """
    Шаг [10, 11] при refractory_until = 10:

        t == refractory_until
        -> нейрон уже свободен, интегрируется весь dt.
    """
    n = LIFNeuron()

    n.refractory_until = 10.0
    n.v = n.v_reset

    current = 0.5 * n.threshold_current
    v_inf = n.steady_state(current)

    expected = (
        v_inf
        + (n.v_reset - v_inf)
        * np.exp(-1.0 / n.tau_m)
    )

    spike = n.step(
        input_current=current,
        t=10.0,
        dt=1.0,
    )

    assert spike is False

    assert n.v == pytest.approx(
        expected,
        rel=1e-13,
        abs=1e-13,
    )


def test_zero_refractory_allows_spikes_on_consecutive_steps():
    """
    refractory = 0 допустим: после спайка нейрон сразу свободен
    и при очень сильном токе может спайкать на каждом шаге.
    """
    n = LIFNeuron(
        refractory=0.0
    )

    current = 100.0 * n.threshold_current

    for i in range(5):
        assert n.step(
            input_current=current,
            t=float(i),
            dt=1.0,
        ) is True

        assert n.refractory_until == pytest.approx(
            i + 1.0
        )

        assert n.is_refractory(i + 1.0) is False


# =====================================================================
# ПОВЕДЕНИЕ ПРИ РАЗНЫХ dt
# =====================================================================


def test_spike_count_stable_for_reasonable_dt():
    """
    При разумных dt число спайков стабильно.

    Это НЕ означает полной независимости от dt:
    времена спайков всё равно дискретизируются по сетке.
    """
    n = LIFNeuron()

    current = 1.5 * n.threshold_current

    fine = _simulate_constant_current(
        dt=0.01,
        duration=200.0,
        current=current,
    )

    coarse = _simulate_constant_current(
        dt=0.5,
        duration=200.0,
        current=current,
    )

    assert abs(
        len(fine) - len(coarse)
    ) <= 1


def test_firing_rate_matches_analytic_lif_prediction():
    """
    Сравнение наблюдаемой частоты с аналитической формулой LIF.

    Для постоянного I > I_th:

        ISI =
            tau_m *
            ln(
                (V_inf - V_reset)
                /
                (V_inf - V_threshold)
            )
            +
            refractory

        f = 1000 / ISI

    где ISI измеряется в мс,
    а f получается в Гц.

    Первый интервал начинается от v_rest,
    поэтому сравнение частоты выполняется с допуском 5%.
    """
    n = LIFNeuron()

    current = 1.5 * n.threshold_current

    v_inf = n.steady_state(current)

    membrane_interval = (
        n.tau_m
        * np.log(
            (v_inf - n.v_reset)
            /
            (v_inf - n.v_threshold)
        )
    )

    expected_isi = (
        membrane_interval
        + n.refractory
    )

    expected_rate_hz = (
        1000.0 / expected_isi
    )

    spikes = _simulate_constant_current(
        dt=0.01,
        duration=1000.0,
        current=current,
    )

    # 1000 ms = 1 секунда.
    observed_rate_hz = len(spikes) / 1.0

    assert observed_rate_hz == pytest.approx(
        expected_rate_hz,
        rel=0.05,
    )


@pytest.mark.parametrize(
    "dt",
    [0.01, 0.1, 0.3, 0.7, 1.0],
)
def test_isi_matches_analytic_within_one_dt(dt):
    """
    Точный интегратор не вносит ошибки интегрирования —
    остаётся только дискретизация времени регистрации спайка.

    Спайк регистрируется на первой границе шага, не раньше
    истинного пересечения порога, а рефрактерность отсчитывается
    от этой границы. Поэтому для каждого межспайкового интервала:

        ISI_analytic <= ISI_sim < ISI_analytic + dt

    Значения dt = 0.3 и 0.7 намеренно не кратны refractory = 2.0:
    так проверяются шаги, пересекающие конец рефрактерности.
    """
    n = LIFNeuron()
    current = 1.5 * n.threshold_current

    isi_analytic = _analytic_isi(n, current)

    spikes = _simulate_constant_current(
        dt=dt,
        duration=300.0,
        current=current,
    )

    assert len(spikes) >= 5

    isi = np.diff(spikes)

    assert np.all(isi >= isi_analytic - 1e-9)
    assert np.all(isi <= isi_analytic + dt + 1e-9)


# =====================================================================
# СХОДИМОСТЬ ПО dt
# =====================================================================


@pytest.mark.parametrize(
    "dt",
    [1.0, 0.5, 0.1, 0.05, 0.01],
)
def test_spike_time_converges_to_analytic_with_decreasing_dt(dt):
    """
    Точность спайк-тайминга квантуется до dt:
    спайк фиксируется на границе шага, а не в аналитический
    момент пересечения порога.

    Поэтому время первого спайка удовлетворяет:

        t_cross <= t_spike <= t_cross + dt

    При уменьшении dt верхняя граница стягивается к t_cross —
    это и есть сходимость интегратора по шагу.

    Этот тест закрывает вопрос: «устойчив при любом dt» —
    да, но точность спайк-тайминга зависит от dt.
    """
    n = LIFNeuron()
    current = 1.5 * n.threshold_current

    # Аналитическое время достижения порога из v_rest.
    t_cross = _analytic_first_crossing(n, current)

    spikes = _simulate_constant_current(
        dt=dt,
        duration=t_cross + 10.0,
        current=current,
    )

    assert len(spikes) >= 1

    t_first_spike = spikes[0]

    # Нижняя граница: спайк не может произойти раньше
    # истинного пересечения порога.
    assert t_first_spike >= t_cross - 1e-9

    # Верхняя граница: спайк не позже t_cross + dt.
    assert t_first_spike <= t_cross + dt + 1e-9


def test_spike_time_error_decreases_with_dt():
    """
    Проверка, что с уменьшением dt ошибка спайк-тайминга
    не возрастает.

    ВАЖНО: строгое убывание математически НЕ гарантировано —
    ошибка зависит от dt как «пила»: первая граница сетки >=
    t_cross может оказаться на той же точке для разных dt.

    Пример: при t_cross ≈ 21.972 первые границы для dt = 1.0
    и dt = 0.1 совпадают (обе = 22.0), ошибки равны.

    Поэтому:
      - используем ВЛОЖЕННЫЕ сетки (степени двойки) —
        каждая мелкая сетка содержит все точки крупной,
        значит её первая граница >= t_cross не позже;
      - сравниваем НЕСТРОГО (с допуском на float-шум).

    Дополнительно: float-арифметика вида i * dt может давать
    на разных dt одну и ту же точку с точностью до ulp —
    поэтому допуск 1e-12 корректен и не маскирует реальный
    регресс.
    """
    n = LIFNeuron()
    current = 1.5 * n.threshold_current

    t_cross = _analytic_first_crossing(n, current)

    # Вложенные сетки: 1.0, 0.5, 0.25, ..., ~0.002.
    dts = [2.0 ** -k for k in range(10)]

    errors = [
        _simulate_constant_current(
            dt=dt,
            duration=t_cross + 10.0,
            current=current,
        )[0] - t_cross
        for dt in dts
    ]

    # Нестрогое убывание: errors[i+1] <= errors[i] + допуск.
    assert all(
        b <= a + 1e-12
        for a, b in zip(errors, errors[1:])
    )

    # На самом мелком dt ошибка должна быть < 1e-3 мс.
    assert errors[-1] < 1e-3


# =====================================================================
# ВАЛИДАЦИЯ ВХОДНЫХ ДАННЫХ
# =====================================================================


@pytest.mark.parametrize(
    "current",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_invalid_input_current_raises(current):
    """Некорректный ток -> ValueError."""
    n = LIFNeuron()

    with pytest.raises(ValueError):
        n.step(
            input_current=current,
            t=0.0,
            dt=0.1,
        )


@pytest.mark.parametrize(
    "dt",
    [
        0.0,
        -0.1,
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_invalid_dt_raises(dt):
    """Некорректный dt -> ValueError."""
    n = LIFNeuron()

    with pytest.raises(ValueError):
        n.step(
            input_current=0.0,
            t=0.0,
            dt=dt,
        )


@pytest.mark.parametrize(
    "t",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_invalid_time_raises(t):
    """Некорректное время -> ValueError."""
    n = LIFNeuron()

    with pytest.raises(ValueError):
        n.step(
            input_current=0.0,
            t=t,
            dt=0.1,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tau_m": -1.0},
        {"tau_m": 0.0},
        {"tau_m": np.nan},
        {"tau_m": np.inf},

        {"r_m": -1.0},
        {"r_m": 0.0},
        {"r_m": np.nan},
        {"r_m": np.inf},

        {
            "v_rest": -55.0,
            "v_threshold": -70.0,
        },

        {
            "v_reset": -50.0,
            "v_threshold": -55.0,
        },

        # Границы: неравенства строгие.
        {"v_threshold": -70.0},  # == v_rest
        {"v_reset": -55.0},      # == v_threshold

        {"v_rest": np.nan},
        {"v_rest": np.inf},
        {"v_threshold": np.nan},
        {"v_threshold": np.inf},
        {"v_reset": np.nan},
        {"v_reset": -np.inf},

        {"refractory": -1.0},
        {"refractory": np.nan},
        {"refractory": np.inf},
    ],
)
def test_invalid_constructor_params_raise(kwargs):
    """Некорректные параметры конструктора -> ValueError."""
    with pytest.raises(ValueError):
        LIFNeuron(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_current": np.nan},
        {"input_current": np.inf},
        # r_m * I переполняется: V_inf = ±inf.
        {"input_current": 1e307},
        {"input_current": -1e307},
        {"dt": 0.0},
        {"dt": -1.0},
        {"dt": np.nan},
        {"t": np.nan},
        # t + dt переполняется.
        {"t": 1e308, "dt": 1e308},
    ],
)
def test_invalid_step_does_not_change_state(kwargs):
    """
    Ошибка валидации не оставляет нейрон в полуобновлённом
    состоянии.

    Шаг по умолчанию [2, 4] пересекает конец рефрактерности
    (refractory_until = 3): если бы проверка шла после присваиваний,
    v успел бы стать v_reset.
    """
    n = LIFNeuron()

    n.v = -60.0
    n.refractory_until = 3.0

    call = {
        "input_current": 0.1,
        "t": 2.0,
        "dt": 2.0,
    }
    call.update(kwargs)

    with pytest.raises(ValueError):
        n.step(**call)

    assert n.v == -60.0
    assert n.refractory_until == 3.0


def test_valid_boundary_params_are_accepted():
    """refractory = 0 допустим; целые числа приводятся к float."""
    n = LIFNeuron(
        tau_m=20,
        r_m=100,
        v_rest=-70,
        v_threshold=-55,
        v_reset=-75,
        refractory=0,
    )

    assert n.refractory == 0.0

    for value in (
        n.tau_m,
        n.r_m,
        n.v_rest,
        n.v_threshold,
        n.v_reset,
        n.refractory,
    ):
        assert isinstance(value, float)


# =====================================================================
# UTILITY API
# =====================================================================


def test_reset_restores_initial_state():
    """reset() возвращает нейрон в исходное состояние."""
    n = LIFNeuron()

    current = 2.0 * n.threshold_current

    for step in range(100):
        n.step(
            input_current=current,
            t=step * 0.1,
            dt=0.1,
        )

    n.reset()

    assert n.v == pytest.approx(
        n.v_rest
    )

    assert n.refractory_until == -np.inf


def test_reset_clears_refractory_state():
    """reset() снимает активную рефрактерность."""
    n = LIFNeuron()

    n.v = n.v_reset
    n.refractory_until = 100.0

    n.reset()

    assert n.v == pytest.approx(
        n.v_rest
    )

    assert n.is_refractory(0.0) is False


def test_reset_gives_same_spike_train_as_fresh_neuron():
    """После reset() нейрон ведёт себя как новый: те же спайки."""
    current = 1.5 * LIFNeuron().threshold_current

    dt = 0.1
    n_steps = 1000

    def run(neuron):
        times = []

        for i in range(n_steps):
            if neuron.step(
                input_current=current,
                t=i * dt,
                dt=dt,
            ):
                times.append(i * dt + dt)

        return times

    used = LIFNeuron()

    # Портим состояние (мембрана, рефрактерность), затем сбрасываем.
    run(used)
    used.reset()

    assert run(used) == run(LIFNeuron())


@pytest.mark.parametrize(
    "current",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_steady_state_rejects_invalid_current(current):
    """steady_state() тоже валидирует ток."""
    n = LIFNeuron()

    with pytest.raises(ValueError):
        n.steady_state(current)


@pytest.mark.parametrize(
    "t",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_is_refractory_rejects_invalid_time(t):
    """is_refractory() тоже валидирует время."""
    n = LIFNeuron()

    with pytest.raises(ValueError):
        n.is_refractory(t)


def test_repr_runs():
    """__repr__ возвращает информативную строку."""
    n = LIFNeuron()

    s = repr(n)

    assert "LIFNeuron" in s
    assert "v=-70.00 mV" in s
    assert "τ_m" in s
    assert "r_m" in s