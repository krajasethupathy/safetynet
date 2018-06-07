import numpy.random as npr
import numpy as np
import random 

class SafetyNetSimulation():
    def __init__(self, 
                 is_high_demand=False, 
                 is_high_truncation=False, 
                 is_high_variable_cancel_rate=False,
                 nCategories = 50, nThresholds = 20, 
                 n_demand_mean = 10, n_demand_stddev = 5,
                 n_inventory_rate_min = 1, n_inventory_rate_max = 20, # inventory rate bounds
                 category_price_mean = 800, category_price_stddev = 400, # price moments
                 a_mean = .5, a_stddev = .1, b_mean = -.2): # cancel params
        
        self.is_high_demand = is_high_demand
        self.is_high_truncation = is_high_truncation
        self.is_high_variable_cancel_rate = is_high_variable_cancel_rate
        
        self.nCategories = nCategories
        self.nThresholds = nThresholds
         
        """
        Each day of the simulation allows the Poisson processes generating 
        demand to run for either 40 or 400 time units (depending on simulation condition)
        """
        if self.is_high_demand:
            self.n_steps = 400
        else:
            self.n_steps = 40
            
        """
        We vary data truncation by setting our benchmark Retail-1-threshold policies to single thresholds
        of either 8 or 14, allowing approximately 1/3 and 2/3 of total demand to get truncated,
        respectively.
        """
        if self.is_high_truncation:
            self.retail_1_threshold_policy = 14
        else:
            self.retail_1_threshold_policy = 8
        
        """
        We control cancel variability by running simulations where the standard deviation
        of cancel parameter b is .05 as well as .15.
        """
        if self.is_high_variable_cancel_rate:
            self.b_stddev = .15
        else:
            self.b_stddev = .05
            

        # Lambdas for Poisson Process generating Demand
        self.category_demand_rate = npr.normal(loc=n_demand_mean, 
                                               scale=n_demand_stddev, 
                                               size=self.nCategories)
        
        self.category_demand_rate = np.clip(self.category_demand_rate, a_min=0, a_max=None)

        # AVG PRICE PER CATEGORY
        self.category_price_mean = category_price_mean
        self.category_price_stddev = category_price_stddev

        # INVENTORY LEVEL
        self.category_inventory_level_rate = npr.uniform(high=n_inventory_rate_max, 
                                                    low=n_inventory_rate_min, 
                                                    size=self.nCategories)
        
        self.category_inventory_level_rate = np.clip(self.category_inventory_level_rate, 
                                                     a_min=0, a_max=None)

        # CANCELLATION PROBABILITY
        self.a_param = npr.normal(loc=a_mean, scale=a_stddev, size=self.nCategories)
        self.b_param = npr.normal(loc=b_mean, scale=self.b_stddev, size=self.nCategories)
        
    def category_demand_to_orders(self, category_demand):
        """
        1 hot encode a category demand (nCategories,) as a matrix with dim (nOrders, nCategories)
        Returned matrix tells us the category for each order
        """
        rows = []
        lc = len(category_demand)
        for i, a in enumerate(category_demand):
            for _ in range(a):
                z = np.zeros(lc)
                z[i] = 1
                rows.append(z)
        if len(rows) > 0:
            return np.stack(rows)
        else:
            return np.array(rows)

    def inv_count_from_orders(self, orders):
        """
        For each order in (nOrders, nCategories), generates an inventory level using a poisson process 
        and 1-hot encodes the inventory at the time of the order in a (nOrders, nThresholds) matrix
        If inventory exceeds nThresholds, cap it at nThreshold 
        [Not sure if this is the proper way to handle it]
        """
        inv_count = []
        for o in orders:
            category_ind = np.argmax(o)
            inv_count_row = np.zeros(shape=(self.nThresholds,))
            inv_level = npr.poisson(self.category_inventory_level_rate[category_ind])
            if inv_level >= self.nThresholds:
                inv_level = self.nThresholds-1
            inv_count_row[inv_level] = 1
            inv_count.append(inv_count_row)

        return np.array(inv_count)

    def price_from_orders(self, orders):
        """
        For each order, generate a price. 
        Returns a (nOrders,) vector
        """
        return npr.normal(loc=self.category_price_mean, 
                          scale=self.category_price_stddev, 
                          size=len(orders))

    def cancel_prob(self, orders, inv_count):
        """
        For each order, calculate a cancellation probability according to that categories params
        returns a (nOrders,) vector
        """
        order_cat = np.argmax(orders, axis= 1) # get category for each order
        inv_cat = np.argmax(inv_count, axis=1) # get inventory for each order
        a_param_all = [self.a_param[i] for i in order_cat]
        b_param_all = [self.b_param[i] for i in order_cat]
        cancel_probs = 1.0 - 1.0/(1.0+ np.exp(a_param_all + (b_param_all * inv_cat)))
        return cancel_probs
    
    def generate_data(self):
        
        X_T = [np.random.poisson(np.clip(rate, a_min=0, a_max=None) , size=self.n_steps) 
               for rate in self.category_demand_rate]
        category_demand = np.array([np.sum(X) for X in X_T])
             
        orders = self.category_demand_to_orders(category_demand)
        np.random.shuffle(orders)

        inv_count = self.inv_count_from_orders(orders)
        price = self.price_from_orders(orders)
        cancel_probs = self.cancel_prob(orders, inv_count)

        collection_thresholds = np.zeros(shape=(orders.shape[0],self.nThresholds))
        collection_thresholds[:,self.retail_1_threshold_policy]=1
        
        # Some sanity checks
        assert orders.shape == (orders.shape[0], self.nCategories)
        assert inv_count.shape == (orders.shape[0], self.nThresholds)
        assert price.shape == (orders.shape[0],)
        assert cancel_probs.shape == (orders.shape[0],)
        assert collection_thresholds.shape == (orders.shape[0],self.nThresholds)
        
        
        return orders, inv_count, price, cancel_probs, collection_thresholds
    

            