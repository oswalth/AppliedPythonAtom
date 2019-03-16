#!/usr/bin/env python
# coding: utf-8


class TEventStats:
    FIVE_MIN = 300

    def __init__(self):
        # TODO: реализовать метод
        self.activity = {}

    def register_event(self, user_id, time):
        """
        Этот метод регистрирует событие активности пользователя.
        :param user_id: идентификатор пользователя
        :param time: время (timestamp)
        :return: None
        """
        # TODO: реализовать метод
        if not self.activity.get(user_id):
            self.activity[user_id] = [time]
        else:
            self.activity[user_id].append(time)

    def query(self, count, time):
        """
        Этот метод отвечает на запросы.
        Возвращает количество пользователей, которые за последние 5 минут
        (на полуинтервале времени (time - 5 min, time]), совершили ровно count действий
        :param count: количество действий
        :param time: время для рассчета интервала
        :return: activity_count: int
        """
        # TODO: реализовать метод
        active_users = []
        for user, timestamp in self.activity.items():
            cnt = 0
            for t in timestamp:
                if 0 <= time - t < self.FIVE_MIN:
                    cnt += 1
            if not cnt and not count:
                continue
            if cnt == count:
                active_users.append(user)

        return len(active_users)
