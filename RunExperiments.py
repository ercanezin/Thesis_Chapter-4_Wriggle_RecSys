import json
import math
import warnings
from datetime import timedelta
from enum import Enum
from math import cos, asin, sqrt

import numpy as np
import pandas as pd
import pytz
from bson import ObjectId
from fastread import Fastread
from sklearn.metrics.pairwise import cosine_similarity


from multiprocessing import Pool, cpu_count
from datetime import datetime
import traceback


from Models import *

warnings.filterwarnings("ignore")


class CITY(Enum):
    Bristol = ObjectId('53e0fa9fd1bd63ee6c56f503')
    Brighton = ObjectId('569624a3fc0b2944dee54535')
    Cardiff = ObjectId('57b58e9f5f384c3898db6b16')


DINNER_DISTANCE = 1.65  # Average in kilometers
LUNCH_DISTANCE = 1.44  # Average in kilometers
AVG_DISTANCE = (DINNER_DISTANCE + LUNCH_DISTANCE) / 2


def get_offers_from_schedule_occurrences(loc, user_id_str, receipt):
    loc_lat = float(loc["lat"])
    loc_lng = float(loc["lng"])
    timezone = pytz.timezone('Europe/London')
    dt_now = receipt['date']
    next_day = dt_now + timedelta(days=1)
    midnight = datetime(next_day.year, next_day.month, next_day.day, tzinfo=timezone)

    filter_criteria = {'cityId': receipt['cityId'], '$and': [
        {"claimEnd": {"$gt": dt_now}},
        {"claimEnd": {"$lt": midnight}},
        {"stockAvailable": {"$gt": 0}}
    ]}

    offer_ids_in_schedule_occurrences = list(ScheduleOccurrences.objects.only("offerId")(__raw__=filter_criteria))

    offers = list()
    for offer_id in offer_ids_in_schedule_occurrences:
        offer = Offers.objects(id=offer_id["offerId"]).first()
        if 'price' in offer:
            offers.append(offer)

    nearby_offers = list()
    offer_dicts = dict()
    distances = dict()
    all_categories = list()
    offer_categories = dict()
    business_ratings = dict()

    user = Users.objects(id=ObjectId(user_id_str)).first()
    user_diet = []
    if user:
        user_diet = user['dietary']  # in case user is removed, we still want to consider recommendation

    diet_veggie_reference = ['vegetarian', 'vegan']

    is_user_diet_selected = any(veg in diet_veggie_reference for veg in user_diet)

    for off in offers:
        # check to make sure diet offer is considered during offer retrieval
        if is_user_diet_selected:
            offer_diet = off['dietaryOptions']
            is_offer_contain_diet = any(veg in diet_veggie_reference for veg in offer_diet)
            if not is_offer_contain_diet:
                continue

        offer_dist = calculate_distance(loc_lat, loc_lng,
                                        off._location["coordinates"][1],
                                        off._location["coordinates"][0])

        if offer_dist <= AVG_DISTANCE:
            nearby_offers.append(off)
            distances[str(off["id"])] = offer_dist
            offer_categories[str(off["id"])] = off["categories"]
            all_categories = list(off["categories"]) + all_categories
            business_ratings[off["id"]] = Businesses.objects(id=off["business"]["_id"]).first()["rating"]

    # Normalising distance values
    if len(nearby_offers) != 0:
        furthest_dist = max(distances.values())

        for offer_id, distance in distances.items():
            distances[offer_id] = 1 - (distance / furthest_dist)

    offer_dicts["nearby_offers"] = nearby_offers
    offer_dicts["off_distances"] = distances
    offer_dicts["all_categories"] = list(set(all_categories))
    offer_dicts["off_categories"] = offer_categories
    offer_dicts["bus_ratings"] = business_ratings

    return offer_dicts


def get_offer_ids_from_recsys(user_id_str, offer_dicts, parameter, receipt_ids_list):
    user_receipts = Receipts.objects(_id__in=[ObjectId(receipt_id) for receipt_id in receipt_ids_list]).only('offerId',
                                                                                                             'amount')

    receipt_offer_ids = list(set([receipt["offerId"] for receipt in user_receipts]))

    avg_user_budget = sum([float(receipt["amount"]) for receipt in user_receipts]) / len(receipt_ids_list)

    offers_by_receipt = Offers.objects(_id__in=receipt_offer_ids)

    user_categories = [category for off in offers_by_receipt for category in off["categories"]]

    scheduled_off_cats = offer_dicts["off_categories"]
    scheduled_off_cats["user_vect"] = list(set(user_categories))
    offer_dicts["off_categories"] = scheduled_off_cats

    offer_dicts["all_categories"].extend(set(user_categories) - set(offer_dicts["all_categories"]))
    recommendations_offer_list = get_recommendations(user_id_str, offer_dicts, receipt_ids_list, avg_user_budget,
                                                     parameter)

    return recommendations_offer_list


def calculate_distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    a = 0.5 - cos((lat2 - lat1) * p) / 2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    return 12742 * asin(sqrt(a))


def get_recommendations(user_id, off_dicts, user_receipts, avg_user_budget, parameter):
    user_cat_sim_dict = calculate_user_category_similarity(off_dicts)

    user_vec = [user_id, 1, 1, 1, 1]

    # off_sim = calculate_user_offer_similarity(user, cat_sim, off_dicts, user_receipts,avg_user_budget)
    off_sim = matching(user_vec, user_cat_sim_dict, off_dicts, user_receipts, avg_user_budget, parameter)

    return off_sim


def calculate_user_category_similarity(off_dicts):
    # Create a numpy matrix that has all categories as columns and all offers as rows including user vector.
    # Initialize a DataFrame with zeros, where rows represent offers and columns represent categories.
    # Initialize all entries as boolean values.
    all_categories = off_dicts["all_categories"]
    mx = pd.DataFrame(0, index=list(off_dicts["off_categories"].keys()), columns=all_categories, dtype='bool')

    off_categories = off_dicts["off_categories"]

    # Iterate over each row (offer) in the matrix and each category in the corresponding offer's category list.
    # Set the value to 1 if the category is present in the offer's category list.
    for index, row in mx.iterrows():
        cat_list = off_categories[index]
        for cat in cat_list:
            row[cat] = 1

    # Move the last row to the top to represent the user vector.
    rows = mx.index.tolist()
    rows = rows[-1:] + rows[:-1]
    mx = mx.loc[rows]

    # Convert the DataFrame to a numpy array for cosine similarity calculation.
    all_data = mx.values

    # Calculate the cosine similarity between the user vector (first row) and all other rows (offers) in the matrix.
    cos_sim = cosine_similarity(all_data[0:1], all_data)[0]

    # Create a dictionary mapping offer indices to their cosine similarity scores with the user vector.
    sim_dict = dict(zip(mx.index.tolist(), cos_sim))

    return sim_dict


def matching(user, cat_sim, off_dicts, user_receipts, avg_user_budget, parameter):
    cols = ['Id', 'distance', 'budget', 'categories', 'rating']
    distances = off_dicts["off_distances"]  # scheduled_offers
    offer_list = list()
    bus_ratings = off_dicts["bus_ratings"]

    for off in off_dicts["nearby_offers"]:
        offer_id = off["id"]

        if float(off["price"]) <= avg_user_budget:
            off_price = 1
        else:
            off_price = avg_user_budget / float(off["price"])  # normalising oof price

        if bus_ratings[offer_id] is None:
            rating = 0
        else:
            rating = bus_ratings[offer_id] / 10  # max rating by wriggle to normalise

        offer_list.append([offer_id,
                           distances[str(offer_id)],
                           off_price,
                           0 if cat_sim is None else cat_sim[str(offer_id)],
                           rating])

    mx = pd.DataFrame(offer_list, columns=cols)

    history_length = len(user_receipts)

    user_vector = pd.DataFrame([user], columns=cols)

    off_sim = dict()

    for index, offer_vector in mx.iterrows():
        # Parameters to play with. Alpha can move between 0 and alpha_max. Beta can only be either 0 or beta_max
        alpha_max = 0.4
        beta_max = 0.3

        sim_alpha = 2.0 - np.sqrt(np.power(user_vector['budget'] - offer_vector['budget'], 2) +
                                  np.power(user_vector['categories'] - offer_vector['categories'], 2))

        # Calculation for optimised
        if parameter and parameter[0] == 0 and parameter[1] == 0 and parameter[2] == 0:

            alpha = min(history_length / 10.0, alpha_max)

            # Determine beta for the current restaurant, depending on whether there is (rating) information or not

            if pd.isnull(offer_vector['rating']):
                beta = 0.0
            else:
                beta = beta_max

            # Determine gamma for the location and distance, based on alpha and beta
            gamma = 1 - (alpha + beta)

            # Similarity based on the business reputation category: rating
            if beta != 0.0:
                sim_beta = 1.0 - np.abs(user_vector['rating'] - offer_vector['rating'])
            else:
                sim_beta = 0.0

            # Similarity based on distance between user location and offer: distance
            sim_gamma = 1.0 - np.abs(user_vector['distance'] - offer_vector['distance'])

            # Finally, calculate the global user-offer matching as the weighted average of the three similarity, using
            # alpha, beta and gamma as weights.

            off_sim[str(offer_vector['Id'])] = float(
                ((alpha * sim_alpha) + (beta * sim_beta) + (gamma * sim_gamma)))
        else:

            alpha = parameter[0]

            # Determine beta for the current restaurant, depending on whether there is (rating) information or not
            beta = parameter[1]

            # Determine gamma for the location and distance, based on alpha and beta
            gamma = parameter[2]

            # Similarity based on the business reputation category: rating
            if beta != 0.0:
                sim_beta = 1.0 - np.abs(user_vector['rating'] - offer_vector['rating'])
            else:
                sim_beta = 0.0

            # Similarity based on distance between user location and offer: distance
            sim_gamma = 1.0 - np.abs(user_vector['distance'] - offer_vector['distance'])

            off_sim[str(offer_vector['Id'])] = float(
                ((alpha * sim_alpha) + (beta * sim_beta) + (gamma * sim_gamma)))

    rev = {k: v for k, v in sorted(off_sim.items(), key=lambda x: x[1], reverse=True)}
    return list(rev.keys())


def process_user(user_id_str, receipt_ids_list):
    receipt_object_ids = [ObjectId(receipt_id) for receipt_id in receipt_ids_list]

    # Retrieve the latest receipt making sure lat-lng info exist with location_exist check
    last_receipt = Receipts.objects(id__in=receipt_object_ids, location_exist=True,
                                    userId=ObjectId(user_id_str)).order_by('-date').first()

    user_loc = {'lat': last_receipt['user_lat'], 'lng': last_receipt['user_lng']}

    results = dict()
    try:
        offers_dicts = get_offers_from_schedule_occurrences(user_loc, user_id_str, last_receipt)
        # Check if offer is in the nearby offer list and flag it- Security check by Wriggle
        flag = False
        for offer in offers_dicts['nearby_offers']:
            if str(last_receipt['offerId']) == str(offer['id']):
                flag = True
                break

        if flag:  # Beta=popularity, alfa=content, gamma=distance,
            parameters = [[1, 0, 0],  # most popular
                          [0, 1, 0],  # User_preference
                          [0, 0, 1],  # location
                          [1 / 3, 1 / 3, 1 / 3],  # same_weight
                          [0, 0, 0]]  # Optimised Weight

            results = []
            for parameter in parameters:
                recommended_offer_ids = get_offer_ids_from_recsys(user_id_str, offers_dicts, parameter,
                                                                  receipt_ids_list)
                user_dict = {'Receipt': last_receipt,
                             'Offer_Ids': recommended_offer_ids,
                             'Business_index': get_offer_business_bool_list(recommended_offer_ids, last_receipt)}
                results.append(user_dict)

    except Exception as error:
        traceback_str = traceback.format_exc()
        print(f" Error at process_user() UserID: {user_id_str}, Error Message: {error}\n{traceback_str}")

    return results  # the order of the result is in order of parameters


def process_user_wrapper(args):
    user_id_str, receipt_ids_list = args
    try:
        return process_user(user_id_str, receipt_ids_list)
    except Exception as error:
        traceback_str = traceback.format_exc()
        print(f" Error at process_user() UserID: {user_id_str}, Error Message: {error}\n{traceback_str}")
        return None


def calculate_experiment_parameters(user_rec_dict):
    most_pop = dict()
    user_preference = dict()
    location = dict()
    same_weight = dict()
    optimised_weight = dict()

    num_of_user_to_process = len(user_rec_dict)
    print(f'{datetime.now()}-Started calculating user experiment parameters')

    counter = 0
    param_results = None

    # Determine number of processes based on CPU count
    num_processes = int(cpu_count() * 0.8)
    pool = Pool(processes=num_processes)

    try:
        # Use multiprocessing Pool to process users concurrently
        results = pool.map(process_user_wrapper, user_rec_dict.items())
    finally:
        pool.close()
        pool.join()

    for user_id_str, param_results in zip(user_rec_dict.keys(), results):
        if not param_results:
            continue
        for i, param in enumerate(param_results):
            if i == 0:
                most_pop[user_id_str] = param
            elif i == 1:
                user_preference[user_id_str] = param
            elif i == 2:
                location[user_id_str] = param
            elif i == 3:
                same_weight[user_id_str] = param
            elif i == 4:
                optimised_weight[user_id_str] = param

    print(f'{datetime.now()}-Ended calculating user experiment parameters')
    return [most_pop, user_preference, location, same_weight, optimised_weight]


def calculate_experiment_parameterss(user_rec_dict):
    most_pop = dict()
    user_preference = dict()
    location = dict()
    same_weight = dict()
    optimised_weight = dict()

    num_of_user_to_process = len(user_rec_dict)
    print(f'{datetime.now()}-Started calculating user experiment parameters')

    counter = 0
    param_results = None
    for user_id_str, receipt_ids_list in user_rec_dict.items():
        try:
            param_results = process_user(
                user_id_str,
                receipt_ids_list)
        except Exception as error:
            traceback_str = traceback.format_exc()
            print(f" Error at process_user() UserID: {user_id_str}, Error Message: {error}\n{traceback_str}")
        counter += 1

        print(f'{datetime.now()} - {counter}/{num_of_user_to_process} User processed ')

        for i, param in enumerate(param_results):
            if i == 0:
                most_pop[user_id_str] = param
            if i == 1:
                user_preference[user_id_str] = param
            if i == 2:
                location[user_id_str] = param
            if i == 3:
                same_weight[user_id_str] = param
            if i == 4:
                optimised_weight[user_id_str] = param

    print(f'{datetime.now()}-Ended calculating user experiment parameters')
    return [most_pop, user_preference, location, same_weight, optimised_weight]


def save_experiment_result(recall_list, ndcg_list, city_name, duration, experiment_param_name):
    file_name = f"./experiment_result/experiment_results_{city_name}_{duration}_{experiment_param_name}.csv"
    with open(file_name, 'w') as file:
        file.write(f'{recall_list}\n')
        file.write(f'{ndcg_list}\n')


def calculate_averages(parameter_result_dicts, city_name, duration):
    experiment_param_names = ['Most-Popular', 'User-Preference', 'Location', 'Same-Weight', 'Optimised-Weight']

    for i_experiment, experiment_param in enumerate(parameter_result_dicts):
        ndcg_list = []
        recall_list = []
        for userId, user_dict in experiment_param.items():
            offer_ids = user_dict['Offer_Ids']
            ndcg = 0
            recall = 0
            if offer_ids.index(str(user_dict['Receipt']['offerId'])) < 10:
                recall += 1
            else:
                for item in user_dict['Business_index']:
                    if item < 10:
                        recall += 0.5

            for i in range(10):
                if offer_ids.index(str(user_dict['Receipt']['offerId'])) == i:
                    ndcg += 1 / math.log(i + 2, 2)
                elif i in user_dict['Business_index']:
                    ndcg += (math.sqrt(2) - 1) / math.log(i + 2, 2)
            recall_list.append(recall)
            ndcg_list.append(ndcg)

        save_experiment_result(recall_list, ndcg_list, city_name, duration, experiment_param_names[i_experiment])

        avg_recall = sum(recall_list) / len(experiment_param)
        avg_ndcg = sum(ndcg_list) / len(experiment_param)

        print('METHOD: ', experiment_param_names[i_experiment])
        print('Avg Recall', avg_recall)
        print('Avg NDCG', avg_ndcg)
        print('------------------------------------------------------------------------------')


def get_offer_business_bool_list(recommended_offer_ids, receipt):
    offers = Offers.objects(_id__in=recommended_offer_ids)
    index_of_business_offers = []
    for offer in offers:
        if str(offer['businessId']) == str(receipt['businessId']) and str(offer['id']) != str(receipt['offerId']):
            index_of_business_offers.append(recommended_offer_ids.index(str(offer['id'])))

    return index_of_business_offers


def record_receipt_times():
    ff = Fastread("./export_sparks_purchases.json")

    events = []
    for line in ff.lines():
        try:
            jsonline = json.loads(line)
            if 'User location lat' in jsonline['properties']:
                events.append(jsonline)
        except:
            pass
    print('Line reading done')
    counter = 0
    for event in events:
        user_id = ObjectId(event['properties']['User Id'])
        offer_id = ObjectId(event['properties']['Offer Id'])
        business_id = ObjectId(event['properties']['Business Id'])
        purchase_datetime = float(event['properties']['mp_processing_time_ms']) / 1000
        dt_purchase = datetime.utcfromtimestamp(purchase_datetime)
        dt_purchase_plus = dt_purchase + timedelta(minutes=2)
        dt_purchase_minus = dt_purchase + timedelta(minutes=-2)

        receipts = Receipts.objects(userId=user_id, businessId=business_id, offerId=offer_id).filter(
            date__gte=dt_purchase_minus, date__lte=dt_purchase_plus)

        if len(receipts) == 0:
            receipts = Receipts.objects(userId=user_id, businessId=business_id, offerId=offer_id).filter(
                date__gte=dt_purchase_minus + timedelta(hours=-1), date__lte=dt_purchase_plus + timedelta(hours=-1))

        if len(receipts) > 1:
            print(user_id)
            print(len(receipts))
            print(event['properties'])

        for receipt in receipts:
            receipt['user_lat'] = event['properties']['User location lat']
            receipt['user_lng'] = event['properties']['User location lng']
            receipt['location_exist'] = True
            receipt.save()
            print(receipt['id'])
            counter += 1

    print("Processed Receipts: ", counter)
    return None


def create_user_receipt_dict(user_receipts_str, location_user_ids):
    print(f'{datetime.now()}-Creating User-Receipt Dictionary')
    user_receipt_dict = {str(user_id_obj): [] for user_id_obj in location_user_ids}
    for receipt_id_user_id in json.loads(user_receipts_str):
        receipt_u_id = receipt_id_user_id['userId']['$oid']
        if receipt_u_id in location_user_ids:
            user_receipt_dict[receipt_u_id].append(receipt_id_user_id['_id']['$oid'])

    for user_key in list(user_receipt_dict.keys()):
        if len(user_receipt_dict[user_key]) < 4:
            del user_receipt_dict[user_key]

    print(f'{datetime.now()}-Ended User-Receipt Dictionary')
    return user_receipt_dict


def load_user_ids_with_location_exist(city_name):
    with open('fast_load/' + city_name + '_user_ids_with_location_exist.csv', 'r', encoding="utf8") as f:
        user_ids_str = f.read()
        user_ids_list = user_ids_str.split(',')
    return user_ids_list


def save_user_ids_with_location_exist(user_ids, city_name):
    user_ids_str = [str(user_id) for user_id in user_ids]
    with open(f'fast_load/{city_name}_user_ids_with_location_exist.csv', 'w+', encoding="utf8") as f:
        f.write(','.join(user_ids_str))


def main():
    # We used the method below to add location information collected from mobile app.between 12 December and 17 May
    # All location information is already added. We keep this for reference. receipts.get(userId=location_user_ids[1])
    # record_receipt_times() #--> num of Receipt  with location_exist information: 56269  Distinct User Number: 14549
    # Receipt num distribution Bristol: 40159 Brighton: 10900 Cardiff: 5210
    # Distinct user number distribution:  Bristol: 14567 Brighton: 3635 Cardiff: 2367
    # return None

    experiment_start_time = datetime.now()

    # First record of location_exist= true date : 1544777907 - 2018-12-14T11:58:28.546Z
    # To make sure receipt data is dependable, we added 1 week margin to the end of the receipts.
    last_receipt_date = datetime.fromtimestamp(1558082829)  # unix timestamp: 1558082829- 2019-05-17T11:47:08.479Z
    last_receipt_date_adjusted = last_receipt_date - timedelta(days=7)
    six_month_from_last = last_receipt_date_adjusted - timedelta(days=182, hours=15)  # 6 months earlier
    one_year_from_last = last_receipt_date_adjusted - timedelta(days=365, hours=6)  # 12 months earlier
    app_launch_date = datetime.fromtimestamp(1391990400)  # 12 feb 2014 -> Wriggle begins operations
    # First record of location_exist= true date : 1544777907 - 2018-12-14T11:58:28.546Z

    experiment_dates = {'6-months': six_month_from_last,
                        '12-months': one_year_from_last,
                        'Since-Beginning': app_launch_date}

    for city in CITY:
        city_start_time = datetime.now()
        # get user_id list that has location information
        print(f'{datetime.now()}-------------{city.name} STARTED')
        # Code below saves user_ids so that we don't have to load every time.The code overrides later if users receipt>3
        # location_user_ids = Receipts.objects(location_exist=True, cityId=ObjectId(city.value)).filter(
        #   date__lte=last_receipt_date_adjusted).distinct('userId')
        # save_user_ids_with_location_exist(location_user_ids, city.name)
        location_exist_user_ids = load_user_ids_with_location_exist(city.name)
        print(f'{datetime.now()}-Num of users with more than 3 receipts: {len(location_exist_user_ids)}')
        for experiment_start_date, start_timestamp in experiment_dates.items():
            print(f'{datetime.now()}-Started Experiment for Time span: {experiment_start_date}')
            receipts_str = Receipts.objects().filter(date__gte=start_timestamp,
                                                     date__lte=last_receipt_date_adjusted).only('_id',
                                                                                                'userId').to_json()

            user_receipt_dict = create_user_receipt_dict(receipts_str, location_exist_user_ids)
            # Update user_ids to reduce user_ids with no more than 3 purchases and
            # use the sames user for all experiment periods.
            # save_user_ids_with_location_exist(list(user_receipt_dict.keys()), city.name)

            parameter_results_list = calculate_experiment_parameters(user_receipt_dict)
            calculate_averages(parameter_results_list, city.name, experiment_start_date)

        print(f'{datetime.now()}-------------{city.name} ENDED- Duration: {datetime.now() - city_start_time}')
    print(f'-----------------------------EXPERIMENT ENDED-----Duration: {datetime.now() - experiment_start_time}\n')


if __name__ == '__main__':
    main()
