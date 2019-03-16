#!/usr/bin/env python
# coding: utf-8


def revert_linked_list(head):
    """
    A -> B -> C should become: C -> B -> A
    :param head: LLNode
    :return: new_head: LLNode
    """
    # TODO: реализовать функцию
    if not head:
        return None
    prev = None
    curr = head
    nextt = None
    while curr:
        nextt = curr.next_node
        curr.next_node = prev
        prev = curr
        curr = nextt
    return prev
