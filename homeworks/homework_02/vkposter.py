#!/usr/bin/env python
# coding: utf-8


# from homeworks.homework_02.heap import MaxHeap
# from homeworks.homework_02.fastmerger import FastSortedListMerger

class User:

    def __init__(self, user_id):
            self.reading = []
            self.posted_posts = []


class Post:

    def __init__(self, post_id):
        self.views = []


class VKPoster:

    def __init__(self):
        self.users = {}
        self.posts = {}
    
    def is_user(self, user_id):
        if user_id not in self.users:
            self.users[user_id] = User(user_id)
    
    def is_post(self, post_id):
        if post_id not in self.posts:
            self.posts[post_id] = Post(post_id)

    def user_posted_post(self, user_id: int, post_id: int):
        '''
        Метод который вызывается когда пользователь user_id
        выложил пост post_id.
        :param user_id: id пользователя. Число.
        :param post_id: id поста. Число.
        :return: ничего
        '''
        self.is_user(user_id)
        self.is_post(post_id)
        self.users[user_id].posted_posts.append(post_id)

    def user_read_post(self, user_id: int, post_id: int):
        '''
        Метод который вызывается когда пользователь user_id
        прочитал пост post_id.
        :param user_id: id пользователя. Число.
        :param post_id: id поста. Число.
        :return: ничего
        '''
        self.is_user(user_id)
        self.is_post(post_id)
        if user_id not in self.posts[post_id].views:
            self.posts[post_id].views.append(user_id)

    def user_follow_for(self, follower_user_id: int, followee_user_id: int):
        '''
        Метод который вызывается когда пользователь follower_user_id
        подписался на пользователя followee_user_id.
        :param follower_user_id: id пользователя. Число.
        :param followee_user_id: id пользователя. Число.
        :return: ничего
        '''
        self.is_user(follower_user_id)
        self.is_user(followee_user_id)
        self.users[follower_user_id].reading.append(followee_user_id)

    def get_recent_posts(self, user_id: int, k: int)-> list:
        '''
        Метод который вызывается когда пользователь user_id
        запрашивает k свежих постов людей на которых он подписан.
        :param user_id: id пользователя. Число.
        :param k: Сколько самых свежих постов необходимо вывести. Число.
        :return: Список из post_id размером К из свежих постов в
        ленте пользователя. list
        '''
        posts = []
        for followee in self.users[user_id].reading:
            for post in self.users[followee].posted_posts:
                posts.append(post)
        return sorted(posts, reverse=True)[:k]

    def get_most_popular_posts(self, k: int) -> list:
        '''
        Метод который возвращает список k самых популярных постов за все время,
        остортированных по свежести.
        :param k: Сколько самых свежих популярных постов
        необходимо вывести. Число.
        :return: Список из post_id размером К из популярных постов. list
        '''
        posts = sorted(self.posts, reverse=True)
        last_populars = sorted(posts, key=lambda post_id: len(self.posts[post_id].views),
                               reverse=True)[:k]
        return last_populars
