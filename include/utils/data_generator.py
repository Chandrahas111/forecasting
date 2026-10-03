import pandas as pd
import numpy as np
import holidays
import os
from typing import Dict, List
from datetime import datetime, timedelta
import random
import logging

logger = logging.getLogger(__name__)


class RealisticSalesDataGenerator :

    def __init__(self, start_date = "2021-01-01", end_date = "2021-12-31"):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.us_holidays = holidays.US()

        self.stores = {
            'store_001' : {'location' : 'New York', 'size' : 'large', 'base_traffic' : 1000},
            'store_002' : {'location' : 'Los Angeles', 'size' : 'large', 'base_traffic' : 950},
            'store_003' : {'location' : 'Chicago', 'size' : 'medium', 'base_traffic' : 800},
            'store_004' : {'location' : 'Houston', 'size' : 'medium', 'base_traffic' : 650},
            'store_005' : {'location' : 'Phoeinx', 'size' : 'small', 'base_traffic' : 400},
            'store_006' : {'location' : 'Philadelphia', 'size' : 'medium', 'base_traffic' : 600},
            'store_007' : {'location' : 'San Antonia', 'size' : 'small', 'base_traffic' : 350},
            'store_008' : {'location' : 'San Diego', 'size' : 'medium', 'base_traffic' : 550},
            'store_009' : {'location' : 'Dallas', 'size' : 'large', 'base_traffic' : 850},
            'store_010' : {'location' : 'Miami', 'size' : 'medium', 'base_traffic' : 600}
        }

        # Product categories and items
        self.product_categories = {
            'Electronics': {
                'ELEC_001': {'name': 'Smartphone', 'price': 699, 'margin': 0.15, 'seasonality': 'holiday'},
                'ELEC_002': {'name': 'Laptop', 'price': 999, 'margin': 0.12, 'seasonality': 'back_to_school'},
                'ELEC_003': {'name': 'Headphones', 'price': 199, 'margin': 0.25, 'seasonality': 'holiday'},
                'ELEC_004': {'name': 'Tablet', 'price': 499, 'margin': 0.18, 'seasonality': 'holiday'},
                'ELEC_005': {'name': 'Smart Watch', 'price': 299, 'margin': 0.20, 'seasonality': 'fitness'}
            },
            'Clothing': {
                'CLTH_001': {'name': 'T-Shirt', 'price': 29, 'margin': 0.50, 'seasonality': 'summer'},
                'CLTH_002': {'name': 'Jeans', 'price': 79, 'margin': 0.45, 'seasonality': 'all_year'},
                'CLTH_003': {'name': 'Jacket', 'price': 149, 'margin': 0.40, 'seasonality': 'winter'},
                'CLTH_004': {'name': 'Dress', 'price': 89, 'margin': 0.48, 'seasonality': 'summer'},
                'CLTH_005': {'name': 'Shoes', 'price': 119, 'margin': 0.42, 'seasonality': 'all_year'}
            },
            'Home': {
                'HOME_001': {'name': 'Coffee Maker', 'price': 79, 'margin': 0.30, 'seasonality': 'holiday'},
                'HOME_002': {'name': 'Blender', 'price': 49, 'margin': 0.35, 'seasonality': 'summer'},
                'HOME_003': {'name': 'Vacuum Cleaner', 'price': 199, 'margin': 0.28, 'seasonality': 'spring'},
                'HOME_004': {'name': 'Air Purifier', 'price': 149, 'margin': 0.32, 'seasonality': 'all_year'},
                'HOME_005': {'name': 'Toaster', 'price': 39, 'margin': 0.40, 'seasonality': 'holiday'}
            },
            'Sports': {
                'SPRT_001': {'name': 'Yoga Mat', 'price': 29, 'margin': 0.55, 'seasonality': 'fitness'},
                'SPRT_002': {'name': 'Dumbbells', 'price': 49, 'margin': 0.45, 'seasonality': 'fitness'},
                'SPRT_003': {'name': 'Running Shoes', 'price': 129, 'margin': 0.38, 'seasonality': 'spring'},
                'SPRT_004': {'name': 'Bicycle', 'price': 399, 'margin': 0.25, 'seasonality': 'summer'},
                'SPRT_005': {'name': 'Tennis Racket', 'price': 89, 'margin': 0.35, 'seasonality': 'summer'}
            }
        }

        self.all_products = {}

        for category, products in self.product_categories.items():
            for product_id, product_info in products.items():
                self.all_products[product_id] = {
                    **product_info,
                    'category' : category
                }

    def get_day_of_week_factor(self, date: pd.Timestamp) -> float:
        dow = date.dayofweek
        # Monday=0, Sunday=6
        dow_factors = [0.9, 0.85, 0.85, 0.9, 1.1, 1.3, 1.2]

        return dow_factors[dow]

    def generate_promotions(self) ->pd.DataFrame:
        # generate random promotions for products
        promotions = []

        major_events = [
            ("Black Friday", 11, 4, 5, 0.25), # 4th friday november , 5 days, 25% off
            ('Cyber Monday', 11, 4, 2, 0.20), #monday after black friday, 2 days, 20% off
            ('Christmas Sale', 12, 15, 10, 0.15), # 15 dec 10 days 15% off
            ('New Year Sale', 1, 1, 7, 0.2), # 1st jan, 7 days, 20% off
            ('Presidents Day', 2, 15, 3, 0.15), # 15 feb, 3 days, 15% off
            ('independance Day Sale', 8, 15, 3, 0.2), 
            ('Labor Day Sale', 9, 1, 3, 0.15),
            ('Halloween Sale', 10, 31, 5, 0.25) # 5 days, 25% off
        ]

        current_date = self.start_date
        while current_date <= self.end_date:
            year = current_date.year

            for event_name, month, day, duration, discount in major_events:
                if event_name == 'Black Friday':
                    november = pd.Timestamp(year, 11, 1)
                    thursdays = pd.date_range(november, november+timedelta(30), freq = 'W-THU')
                    event_date = thursdays[3] + timedelta(days=1) # 4th thursdays of november
                else:
                    try:
                        event_date = pd.Timestamp(year, month, day)
                    except ValueError:
                        continue
                if self.start_date <= event_date <= self.end_date:
                    for d in range(duration):
                        promo_date = event_date + timedelta(days=d)
                        if promo_date <= self.end_date:
                            promo_products = random.sample(list(self.all_products.keys()), k = random.randint(5,15))

                            for product_id in promo_products:
                                promotions.append({
                                    'product_id' : product_id,
                                    'date' : promo_date,
                                    'discount_percent' : discount,
                                    'promotion_type' : event_name
                                })

            current_date = current_date + pd.DateOffset(years=1)

        n_flash_sales = int((self.end_date-self.start_date).days * 0.05)  # # 5% of the time
        flash_dates = pd.date_range(self.start_date,self.end_date, periods=n_flash_sales)

        for date in flash_dates:
            promo_products = random.sample(list(self.all_products.keys()), k = random.randint(3,8))

            for product_id in promo_products:
                promotions.append({
                    'product_id' : product_id,
                    'date' : promo_date,
                    'discount_percent' : random.uniform(0.1, 0.3),  # 10 to 30 % discount
                    'promotion_type' : event_name
                })

        return pd.DataFrame(promotions)
    

    def generate_store_events(self) -> pd.DataFrame:
        """Generate store-specific events (closures, renovations, etc.)"""
        events = []
        
        for store_id, store_info in self.stores.items():
            # Random store closures (weather, technical issues)
            n_closures = random.randint(2, 5)
            closure_dates = pd.date_range(self.start_date, self.end_date, periods=n_closures)
            
            for date in closure_dates:
                events.append({
                    'store_id': store_id,
                    'date': date,
                    'event_type': 'closure',
                    'impact': -1.0  # 100% reduction
                })
            
            # Store renovations (longer impact)
            if random.random() < 0.3:  # 30% chance of renovation
                renovation_start = self.start_date + timedelta(days=random.randint(100, 600))
                renovation_duration = random.randint(7, 21)
                
                for d in range(renovation_duration):
                    reno_date = renovation_start + timedelta(days=d)
                    if reno_date <= self.end_date:
                        events.append({
                            'store_id': store_id,
                            'date': reno_date,
                            'event_type': 'renovation',
                            'impact': -0.3  # 30% reduction
                        })
        
        return pd.DataFrame(events)
    
                        



    def generate_sales_data(self, output_dir : str = '/tmp/sales_data') -> Dict[str, List[str]]:

        os.makedirs(output_dir, exist_ok = True)

        # generate suplimentry data
        promotions_df = self.generate_promotions()
        store_events_df = self.generate_store_events()

        file_paths = {
            'sales' :[],
            'inventory' : [],
            'customer_traffic' : [],
            'promotions' : [],
            'store_events' : []
        }

        # supplementary data
        promotions_path = os.path.join(output_dir, 'promotions/promotions.parquet')
        os.makedirs(os.path.dirname(promotions_path), exist_ok=True)
        promotions_df.to_parquet(promotions_path, index=False)
        file_paths['promotions'].append(promotions_path)

        events_path = os.path.join(output_dir, "store_events/events.parquet")
        os.makedirs(os.path.dirname(events_path), exist_ok=True)
        store_events_df.to_parquet(events_path, index=False)
        file_paths['store_events'].append(events_path)

        # generate sales data by day
        current_date = self.start_date

        while current_date < self.end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            logger.info(f"generating_sales_data for {date_str}")

            # daily_sales_data for all stores
            daily_sales_data = []
            daily_traffic_data = []
            daily_inventory_data = []

            for store_id, store_info in self.stores.items():
                base_traffic = store_info['base_traffic']

                # dates factor
                dow_factor = self.get_day_of_week_factor(current_date)
                is_holiday = current_date in self.us_holidays
                holiday_factor = 1.3 if is_holiday else 1.0

                # weather impact(random)
                weather_factor = np.random.normal(1.0, 0.1)
                weather_factor = max(0.5, min(1.2, weather_factor))

                # check for store events
                store_event_impact = 1.0
                if not store_events_df.empty:
                    event = store_events_df[
                        (store_events_df['store_id'] == store_id) & 
                        (store_events_df['date'] == current_date)
                    ]

                    if not event.empty:
                        store_event_impact = event.iloc[0]['impact'] + 1.0

                # calculate store traffic
                store_traffic = int(
                    base_traffic * dow_factor * holiday_factor * weather_factor * store_event_impact * np.random.normal(1.0, 0.05)
                )

                daily_traffic_data.append({
                    'store_id' : store_id,
                    'date' : current_date,
                    'customer_traffic' : store_traffic,
                    'weather_impact' : weather_factor,
                    'holiday' : is_holiday
                })


                # generate product level sales
                for product_id, product_info in self.all_products.items():
                    seasonality_factor = 1.0
                    # product seasonality
                    if product_info['seasonality'] == 'holiday':
                        seasonality_factor = holiday_factor
                    elif product_info['seasonality'] == 'back_to_school':
                        seasonality_factor = dow_factor
                    elif product_info['seasonality'] == 'spring':
                        seasonality_factor = 1.2 if current_date.month < 6 else 1.0
                    elif product_info['seasonality'] == 'summer':
                        seasonality_factor = 1.2 if current_date.month < 9 else 1.0


                    # check for promotions
                    promotion_factor = 1.0
                    discount_peercent = 0.0
                    if not promotions_df.empty:
                        promo = promotions_df[
                            (promotions_df['date'] == current_date) & 
                            (promotions_df['product_id'] == product_id)
                        ]
                        if not promo.empty:
                            discount_percent = promo.iloc[0]['discount_percent']
                            # Promotion increases demand
                            promotion_factor = 1.0 + (discount_percent * 3)  # 3x multiplier

                    # sales quantity
                    size_factor = {
                        'large' : 1.0,
                        'medium': 0.7,
                        'small' : 0.5
                    }[store_info['size']]
                    price_factor = 1.0/(1.0 + product_info['price']/100.0) # higher price lower volume

                    base_quantity = store_traffic *0.001 * size_factor * price_factor

                    quantity = int(base_quantity*seasonality_factor*promotion_factor*np.random.normal(loc=1.0, scale=0.2))

                    quantity = max(0,quantity)

                    # calculate revenue
                    actual_price = product_info['price'] * (1-discount_peercent)
                    revenue = quantity * actual_price
                    cost = quantity * product_info['price'] * (1- product_info['margin'])
                    
                    if quantity > 0:
                        daily_sales_data.append({
                            'date': current_date,
                            'store_id': store_id,
                            'product_id': product_id,
                            'category': product_info['category'],
                            'quantity_sold': quantity,
                            'unit_price': product_info['price'],
                            'discount_percent': discount_percent,
                            'revenue': revenue,
                            'cost': cost,
                            'profit': revenue - cost
                        })


                    # inventory tracking
                    inventory_level = random.randint(500, 1200)
                    reorder_point = random.randint(20, 50)

                    daily_inventory_data.append({
                        'date': current_date,
                        'store_id': store_id,
                        'product_id': product_id,
                        'inventory_level': inventory_level,
                        'reorder_point': reorder_point,
                        'days_of_supply': inventory_level / max(1, quantity)
                    })

            # save daily file with proper partitioning
            if daily_sales_data:
                sales_df = pd.DataFrame(daily_sales_data)
                sales_path = os.path.join(
                    output_dir, 
                    f"sales/year={current_date.year}/month={current_date.month:02d}/day={current_date.day:02d}/"
                    f"sales_{date_str}.parquet"
                )

                os.makedirs(os.path.dirname(sales_path), exist_ok=True)
                sales_df.to_parquet(sales_path, index=False)
                file_paths['sales'].append(sales_path)

            # customer traffic data
            if daily_traffic_data:
                traffic_df = pd.DataFrame(daily_traffic_data)
                traffic_path = os.path.join(
                    output_dir,
                    f"customer_traffic/year={current_date.year}/month={current_date.month:02d}/day={current_date.day:02d}/"
                    f"traffic_{date_str}.parquet"
                )
                os.makedirs(os.path.dirname(traffic_path), exist_ok=True)
                traffic_df.to_parquet(traffic_path, index=False)
                file_paths['customer_traffic'].append(traffic_path)

            # inventory data
            if daily_traffic_data and current_date.dayofweek ==6 : #only weekly on sundays
                inventory_df = pd.DataFrame(daily_inventory_data)
                inventory_path = os.path.join(
                    output_dir,
                    f"inventory/year={current_date.year}/week={current_date.isocalendar()[1]:02d}/"
                    f"inventory_{date_str}.parquet"
                )
                os.makedirs(os.path.dirname(inventory_path), exist_ok=True)
                inventory_df.to_parquet(inventory_path, index=False)
                file_paths['inventory'].append(inventory_path)

            # move to next day
            current_date = current_date + timedelta(days=1)


        metadata = {
            'generation_date' : datetime.now().isoformat(),
            'start_date' : self.start_date.isoformat(),
            'end_date' : self.end_date.isoformat(),
            'n_stores' : len(self.stores),
            'n_products' : len(self.all_products),
            'file_counts' : {k : len(v) for k,v in file_paths.items()},
            'total_files' : sum(len(v) for v in file_paths.values())
        }

        metadata_df = pd.DataFrame(metadata)
        metadata_path = os.path.join(output_dir, "metadata/generation_metadata.parquet")
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        metadata_df.to_parquet(metadata_path, index=False)


        logger.info(f'generated sales data for {len(file_paths['sales'])} days')
        # logger.info(f'generated {metadata['total_files']} files')
        logger.info(f"Sales files: {len(file_paths['sales'])}")
        logger.info(f"Output directory: {output_dir}")

        return file_paths






