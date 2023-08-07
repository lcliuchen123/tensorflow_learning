# -*- coding: utf-8 -*-

# side_info文件中，列的顺序
feature_names_c = [
    'geek_id', 'geek_gender', 'geek_rev_age', 'five_get_geek_low_salary',
    'geek_school_degree', 'geek_workyears',
    'five_basic_is_overseas', 'geek_apply_status',
    'five_geek_complete_time_diff_day', 'five_get_geek_high_salary',
    'five_get_geek_city_set', 'five_get_geek_position_set',
    'five_get_geek_l2code_set', 'five_get_geek_l1code_set'
    ]

static_array_fea_c = [
    'five_get_geek_city_set', 'five_get_geek_position_set',
    'five_get_geek_l2code_set', 'five_get_geek_l1code_set'
]

static_cate_dict_c = [
    'geek_id', 'geek_gender',
    'geek_rev_age', 'geek_school_degree',
    'five_basic_is_overseas', 'geek_apply_status'
    ]

static_numeric_dict_c = {
    'five_geek_complete_time_diff_day': [5, 26, 83, 169, 307, 604, 945, 1232, 1594],
    'geek_workyears': [0, 1, 2, 4, 6, 8, 11, 15],
    'five_get_geek_low_salary':  [0, 2, 3, 4, 5, 7, 9, 11],
    'five_get_geek_high_salary': [2, 5, 6, 7, 8, 10, 12, 17]
}

feature_name2code_dict = {
    'geek_id': 2673,
    'geek_gender': 204,
    'geek_rev_age': 977,
    'five_get_geek_low_salary': 12005,
    'five_get_geek_high_salary': 12006,
    'geek_school_degree': 210,
    'geek_workyears': 179,
    'five_basic_is_overseas': 423,
    'geek_apply_status': 217,
    'five_geek_complete_time_diff_day': 11934,
    'five_get_geek_city_set': 11935,
    'five_get_geek_position_set': 11938,
    'five_get_geek_l2code_set': 11937,
    'five_get_geek_l1code_set': 11936
}

feature_name2len_dict = {
    'geek_id': 1,
    'geek_gender': 1,
    'geek_rev_age': 1,
    'five_get_geek_low_salary': 1,
    'five_get_geek_high_salary': 1,
    'geek_school_degree': 1,
    'geek_workyears': 1,
    'five_basic_is_overseas': 1,
    'geek_apply_status': 1,
    'five_geek_complete_time_diff_day': 1,
    'five_get_geek_city_set': 3,
    'five_get_geek_position_set': 3,
    'five_get_geek_l2code_set': 3,
    'five_get_geek_l1code_set': 3,
    'labels': 1
}
