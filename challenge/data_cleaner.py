from datetime import timedelta
import datetime

import pandas as pd
import numpy as np
import re

END_FEATURES = ['h_booking_id', 'hotel_age', 'hotel_star_rating', 'guest_is_not_the_customer',
                'no_of_adults', 'no_of_children',
                'no_of_extra_bed', 'no_of_room',
                'original_selling_amount',
                'is_user_logged_in',
                'is_first_booking',
                'days_book_before_checking', 'vacation_duration', 'week_in_year',
                'month_in_year', 'is_weekend', 'num_of_requests',
                'need_to_pay_for_cancellation',
                'original_payment_type_Credit Card', 'original_payment_type_Gift Card',
                'original_payment_type_Invoice', 'charge_option_Pay Later',
                'charge_option_Pay Now', 'charge_option_Pay at Check-in',
                'customer_nationality_cancellation_rate', 'hotel_id_cancellation_rate',
                'hotel_country_code_cancellation_rate',
                'accommadation_type_name_cancellation_rate',
                'origin_country_code_cancellation_rate', 'language_cancellation_rate',
                'hotel_area_code_cancellation_rate',
                'hotel_city_code_cancellation_rate',
                'original_payment_method_cancellation_rate']

TARGET_NAME = 'cancellation_bin'

class DataCleaner:
    def __init__(self):
        self.df = None
        self.is_train = True
        self.mappers = {}

    def _create_target(self):
        # cancelled is 1
        self.target = TARGET_NAME
        self.df[self.target] = ~self.df['cancellation_datetime'].isna()

    def _create_time_period(self):
        """
        create columns:
        days_book_before_checking - days between booking and checking
        vication_duration
        month/week_in_year - the day in the year which the checking start
        weekend binary
        hotel_lived_days
        """
        time_delta_series = pd.to_datetime(self.df['checkin_date']) - pd.to_datetime(self.df['booking_datetime'])
        self.df['days_book_before_checking'] = time_delta_series.dt.days

        self.df['vacation_duration'] = (
                pd.to_datetime(self.df['checkout_date']) - pd.to_datetime(self.df['checkin_date'])).dt.days

        self.df['week_in_year'] = pd.to_datetime(self.df['checkin_date']).dt.isocalendar().week
        self.df['month_in_year'] = pd.to_datetime(self.df['checkin_date']).dt.month
        self.df['is_weekend'] = (pd.to_datetime(self.df['checkin_date']).dt.weekday + self.df['vacation_duration']) >= 4
        self.df['hotel_age'] = (
                    pd.to_datetime(self.df['checkin_date']).max() - pd.to_datetime(self.df['hotel_live_date'])).dt.days

    def _create_num_requests(self):
        requests_features = [col_name for col_name in self.df.columns if 'request' in col_name]
        self.df[requests_features] = self.df[requests_features].fillna(0)
        self.df['num_of_requests'] = sum([self.df[col_name] for col_name in requests_features])

    def _features_from_bool_to_binary(self, features=['is_user_logged_in', 'is_first_booking']):
        for feature in features:
            self.df[feature] = self.df[feature].astype(int)

    def _large_cat_to_groups(self, large_cat_cols=['customer_nationality', 'hotel_id', 'hotel_country_code',
                                                   'accommadation_type_name', 'origin_country_code', 'language',
                                                   'hotel_area_code', 'hotel_city_code', 'original_payment_method'],
                             min_num_apearencess=5):
        if self.is_train:
            for col in large_cat_cols:
                self.df[f'{col}_cancellation_rate'] = self.df[self.target].groupby(self.df[col], sort=False).transform('mean')
                vc = self.df[col].value_counts()    
                bad_values = list(vc[vc <= min_num_apearencess].index)  # not enough appearances
                self.df.loc[[i in bad_values for i in self.df[col]], f'{col}_cancellation_rate'] = 0.5
                gb = self.df[self.target].groupby(self.df[col], sort=False).mean()
                self.mappers[col] = gb.to_dict()
        else:
            for col in large_cat_cols:
                self.df[f'{col}_cancellation_rate'] = self.df[col].map(self.mappers[col])

    def _get_last_free_cancelation_date(self, policy_code, checkin_date) -> datetime:
        NO_SHOW_PATTERN = r"^\d+[PN]$"
        CANCELLATION_PATTERN = r"^\d+D\d+[PN]$"
        policy_periods_list = policy_code.split('_')
        checkin_date = pd.to_datetime(checkin_date)
        days_before = -np.inf
        for policy_period in policy_periods_list:
            if re.match(NO_SHOW_PATTERN, policy_period) or policy_code == "UNKNOWN":
                days_before = max(0, days_before)
            elif re.match(CANCELLATION_PATTERN, policy_period):
                sep_ind = policy_period.index('D')
                days_before = max(int(policy_period[:sep_ind]), days_before)
        return checkin_date - timedelta(days=days_before)

    def _cancellation_pay_train(self):
        self.df['last_free_cancel_date'] = self.df.apply(
            lambda x: self._get_last_free_cancelation_date(x['cancellation_policy_code'], x['checkin_date']), axis=1)

        self.df['cancellation_datetime'].fillna(self.df['checkout_date'])
        self.df['free_period_cancelled'] = pd.to_datetime(self.df['cancellation_datetime']) <= self.df[
            'last_free_cancel_date']
        self.df['paid_period_cancelled'] = (pd.to_datetime(self.df['cancellation_datetime']) > self.df[
            'last_free_cancel_date']) & \
                                           (self.df['last_free_cancel_date'] < self.df['checkout_date'])
        df_free_period = self.df.copy()
        df_paid_period = self.df.copy()
        df_free_period['need_to_pay_for_cancellation'] = 0
        df_paid_period['need_to_pay_for_cancellation'] = 1
        df_paid_period = df_paid_period[df_paid_period['free_period_cancelled'] == True]
        df_free_period[self.target] = self.df['free_period_cancelled']
        df_paid_period[self.target] = self.df['paid_period_cancelled']
        self.df = pd.concat([df_free_period, df_paid_period])

    def _cancellation_pay_test(self):
        self.df['last_free_cancel_date'] = self.df.apply(
            lambda x: self._get_last_free_cancelation_date(x['cancellation_policy_code'], x['checkin_date']), axis=1)
        START_DATE_WINDOW = datetime.datetime(year=2018, month=12, day=7)
        END_DATE_WINDOW = datetime.datetime(year=2018, month=12, day=13)
        df_free_period = self.df.copy()
        df_paid_period = self.df.copy()
        df_free_period['need_to_pay_for_cancellation'] = 0
        df_paid_period['need_to_pay_for_cancellation'] = 1
        df_free_period = df_free_period[df_free_period['last_free_cancel_date'] > START_DATE_WINDOW]
        df_paid_period = df_paid_period[df_paid_period['last_free_cancel_date'] <= END_DATE_WINDOW]
        self.df = pd.concat([df_free_period, df_paid_period])


    def _one_hot_columns(self, features=['original_payment_type', 'charge_option']):
        self.df = pd.get_dummies(self.df, columns=features)

    def _select_specific_features(self, features=END_FEATURES):
        if self.is_train: 
            train_features = features + [self.target]
            self.df = self.df[train_features]
        else:
            for f in features:
                if f not in self.df.columns: self.df[f] = 0 # Add missing one-hot columns
            self.df = self.df[features]

    def _fill_na(self):
        """
        only for *cancellation_rate colls
        """
        columns = [col for col in self.df.columns if 'cancellation_rate' in col]
        self.df[columns] = self.df[columns].fillna(0.5)
        assert not self.df.isna().values.any()

    def run(self, data: pd.DataFrame):
        self.df = data
        self.is_train = 'cancellation_datetime' in data.columns
        if self.is_train:
            self._create_target()
        self._create_time_period()
        self._create_num_requests()
        self._features_from_bool_to_binary()
        self._cancellation_pay_train() if self.is_train else self._cancellation_pay_test()
        self._one_hot_columns()
        self._large_cat_to_groups()
        self._select_specific_features()
        self._fill_na()
        self.df.reset_index()
        return self
