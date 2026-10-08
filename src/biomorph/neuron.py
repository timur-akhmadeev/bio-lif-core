"""
LIF-нейрон (Leaky Integrate-and-Fire) для системы BioMorph.

Модель:
    τ_m · dV/dt = -(V - V_rest) + R_m · I

Единицы:
    время — мс
    напряжение — мВ
    ток — нА
    сопротивление — МОм

    1 МОм · 1 нА = 1 мВ

Интегратор:
    точное аналитическое решение для постоянного тока на шаге.

    При постоянном токе V(t) монотонно стремится к V_inf, поэтому
    максимум потенциала на шаге достигается на одном из его концов.
    Значит, проверка порога в конце шага не пропускает пересечений
    внутри шага — при любом dt.

Устойчивость и точность:
    Интегратор безусловно устойчив при любом dt > 0
    (в отличие от метода Эйлера).

    Однако точность спайк-тайминга квантуется до dt:
    спайк фиксируется на правой границе шага (t + dt),
    а не в аналитический момент пересечения порога.
    При уменьшении dt время спайка сходится к точному
    аналитическому значению.

Рекомендуемый dt: 0.1 мс
    - Для большинства задач этого достаточно.
    - Меньше dt — точнее спайк-тайминг, но медленнее симуляция.
    - Больше dt — быстрее, но грубее времена спайков.
    - Для latency/temporal-кодирования и STDP нужен меньший dt
      (0.01–0.05 мс), поскольку там важны точные времена спайков.

Спайки:
    не хранятся в нейроне — их должен записывать внешний recorder.

Семантика времени:
    step() моделирует интервал [t, t + dt].
    Если порог достигнут на шаге, spike регистрируется
    на правой границе шага: t + dt.

    Это дискретное время регистрации события, а не аналитически
    вычисленное точное время пересечения порога.
"""

import math


class LIFNeuron:
    """
    Один LIF-нейрон.

    Параметры:
        tau_m       — постоянная времени мембраны (мс), > 0
        r_m         — сопротивление мембраны (МОм), > 0
        v_rest      — потенциал покоя (мВ)
        v_threshold — порог спайка (мВ), должен быть > v_rest
        v_reset     — потенциал сброса (мВ), должен быть < v_threshold
        refractory  — абсолютный рефрактерный период (мс), >= 0
    """

    def __init__(
        self,
        tau_m: float = 20.0,
        r_m: float = 100.0,
        v_rest: float = -70.0,
        v_threshold: float = -55.0,
        v_reset: float = -75.0,
        refractory: float = 2.0,
    ):
        # Приводим параметры к float.
        tau_m = float(tau_m)
        r_m = float(r_m)
        v_rest = float(v_rest)
        v_threshold = float(v_threshold)
        v_reset = float(v_reset)
        refractory = float(refractory)

        params = {
            "tau_m": tau_m,
            "r_m": r_m,
            "v_rest": v_rest,
            "v_threshold": v_threshold,
            "v_reset": v_reset,
            "refractory": refractory,
        }

        # Все параметры должны быть конечными.
        for name, value in params.items():
            if not math.isfinite(value):
                raise ValueError(
                    f"{name} должен быть конечным, получено {value}"
                )

        # Физические ограничения.
        if tau_m <= 0.0:
            raise ValueError(
                f"tau_m должен быть > 0, получено {tau_m}"
            )

        if r_m <= 0.0:
            raise ValueError(
                f"r_m должен быть > 0, получено {r_m}"
            )

        if v_threshold <= v_rest:
            raise ValueError(
                f"v_threshold ({v_threshold}) должен быть "
                f"больше v_rest ({v_rest})"
            )

        if v_reset >= v_threshold:
            raise ValueError(
                f"v_reset ({v_reset}) должен быть "
                f"меньше v_threshold ({v_threshold})"
            )

        if refractory < 0.0:
            raise ValueError(
                f"refractory должен быть >= 0, получено {refractory}"
            )

        self.tau_m = tau_m
        self.r_m = r_m
        self.v_rest = v_rest
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.refractory = refractory

        # Динамическое состояние нейрона.
        self.v = v_rest

        # До этого момента нейрон находится в рефрактерности.
        # -inf означает отсутствие активной рефрактерности.
        self.refractory_until = -math.inf

    def __repr__(self) -> str:
        return (
            f"LIFNeuron(v={self.v:.2f} mV, "
            f"τ_m={self.tau_m} ms, "
            f"r_m={self.r_m} MΩ)"
        )

    @property
    def threshold_current(self) -> float:
        """
        Минимальный постоянный ток, при котором
        стационарный потенциал равен порогу:

            I_th = (V_threshold - V_rest) / R_m

        При I < I_th:
            V_inf < V_threshold, порог не достигается.

        При I == I_th:
            V_inf == V_threshold, но из состояния ниже порога
            он достигается только асимптотически, поэтому
            за конечное время спайка не возникает. Это верно
            и для float-арифметики: step() не даёт округлению
            довести V до порога, если V_inf <= V_threshold.

        При I > I_th:
            V_inf > V_threshold, поэтому нейрон может генерировать
            периодические спайки.
        """
        return (
            self.v_threshold - self.v_rest
        ) / self.r_m

    def is_refractory(self, t: float) -> bool:
        """
        Проверяет, находится ли нейрон в рефрактерности
        в момент времени t.

        Граница:
            t < refractory_until  -> refractory
            t >= refractory_until -> свободен
        """
        t = float(t)

        if not math.isfinite(t):
            raise ValueError(
                f"t должен быть конечным, получено {t}"
            )

        return t < self.refractory_until

    def step(
        self,
        input_current: float,
        t: float,
        dt: float,
    ) -> bool:
        """
        Выполняет один шаг симуляции на интервале [t, t + dt].

        Параметры:
            input_current — постоянный ток на этом шаге (нА)
            t             — начало шага (мс)
            dt            — длительность шага (мс), > 0

        Возвращает:
            True  — если на шаге произошёл спайк
            False — если спайка не было

        Исключения:
            ValueError — если входы не конечны, dt <= 0 или
            V_inf = V_rest + R_m · I переполняется. Проверки идут
            до любых изменений состояния: при ошибке нейрон
            остаётся ровно таким, каким был до вызова.

        Семантика времени:
            spike считается зарегистрированным в t + dt.

        Алгоритм:
            1. Проверяем рефрактерность.
            2. Если шаг пересекает конец рефрактерности,
               удерживаем v_reset до refractory_until.
            3. Проверяем, не находится ли v уже на/выше порога.
            4. Выполняем точное аналитическое интегрирование.
            5. Проверяем порог после интегрирования.
            6. При спайке устанавливаем v_reset и начинаем
               новый рефрактерный период.
        """
        input_current = float(input_current)
        t = float(t)
        dt = float(dt)

        # --------------------------------------------------------------
        # Валидация входов
        #
        # Все проверки — до изменения состояния: при ошибке нейрон
        # остаётся таким, каким был до вызова.
        # --------------------------------------------------------------

        if not math.isfinite(input_current):
            raise ValueError(
                "input_current должен быть конечным, "
                f"получено {input_current}"
            )

        if not math.isfinite(t):
            raise ValueError(
                f"t должен быть конечным, получено {t}"
            )

        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError(
                f"dt должен быть конечным > 0, получено {dt}"
            )

        t_end = t + dt

        # Защита от переполнения float при t + dt.
        if not math.isfinite(t_end):
            raise ValueError(
                f"t + dt должен быть конечным, "
                f"получено t={t}, dt={dt}"
            )

        # Стационарный потенциал при постоянном токе
        # (та же формула, что в steady_state()):
        #
        #     V_inf = V_rest + R_m * I
        #
        v_inf = self.v_rest + self.r_m * input_current

        # Защита от переполнения R_m * I: иначе v молча станет NaN,
        # и нейрон навсегда перестанет спайкать.
        if not math.isfinite(v_inf):
            raise ValueError(
                "V_inf = v_rest + r_m * I должен быть конечным, "
                f"получено {v_inf} "
                f"(r_m={self.r_m}, input_current={input_current})"
            )

        # --------------------------------------------------------------
        # 1. Рефрактерность
        # --------------------------------------------------------------

        if t_end <= self.refractory_until:
            # Весь шаг находится внутри рефрактерного периода.
            self.v = self.v_reset
            return False

        if t < self.refractory_until:
            # Шаг пересекает границу рефрактерности.

            # До refractory_until нейрон удерживается на reset.
            self.v = self.v_reset

            # Интегрируем только активную часть шага.
            active_dt = t_end - self.refractory_until

        else:
            # Весь шаг активен.
            active_dt = dt

        # Здесь active_dt > 0 всегда: при t_end <= refractory_until
        # мы бы уже вышли выше.

        # --------------------------------------------------------------
        # 2. Pre-check порога
        # --------------------------------------------------------------

        # После заморозки на v_reset эта проверка заведомо
        # не сработает (по контракту v_reset < v_threshold).
        # Однако она нужна для случая t >= refractory_until,
        # когда v мог быть поднят снаружи до порога или выше.
        #
        # Используем >=, а не ==, чтобы корректно обработать
        # потенциальное превышение порога.
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            self.refractory_until = t_end + self.refractory
            return True

        # --------------------------------------------------------------
        # 3. Точное аналитическое интегрирование
        # --------------------------------------------------------------

        # Exact solution:
        #
        #     V(t + dt) =
        #         V_inf +
        #         (V(t) - V_inf) * exp(-dt / tau_m)
        #
        decay = math.exp(-active_dt / self.tau_m)

        v_new = v_inf + (self.v - v_inf) * decay

        # Защита от ложного спайка из-за округления float.
        #
        # Если V_inf <= V_threshold, а v < V_threshold (это уже
        # гарантировано pre-check'ом), точное решение остаётся строго
        # ниже порога при любом конечном dt.
        #
        # Но при V_inf == V_threshold (I == I_th) и dt порядка
        # tau_m и больше поправка (v - V_inf) * decay оказывается
        # меньше половины ulp, и сумма округляется ровно до
        # V_threshold — то есть получаем спайк, которого в точном
        # решении нет. Возвращаем v на ближайшее float ниже порога.
        if v_inf <= self.v_threshold and v_new >= self.v_threshold:
            v_new = math.nextafter(self.v_threshold, -math.inf)

        self.v = v_new

        # --------------------------------------------------------------
        # 4. Post-check порога
        # --------------------------------------------------------------

        if self.v >= self.v_threshold:
            self.v = self.v_reset
            self.refractory_until = t_end + self.refractory
            return True

        return False

    def steady_state(self, input_current: float) -> float:
        """
        Возвращает аналитический стационарный потенциал:

            V_inf = V_rest + R_m * I
        """
        input_current = float(input_current)

        if not math.isfinite(input_current):
            raise ValueError(
                "input_current должен быть конечным, "
                f"получено {input_current}"
            )

        return self.v_rest + self.r_m * input_current

    def reset(self) -> None:
        """
        Полностью возвращает нейрон в начальное состояние.
        """
        self.v = self.v_rest
        self.refractory_until = -math.inf