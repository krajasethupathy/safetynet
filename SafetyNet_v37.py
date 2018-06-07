class SafetyNet(nn.Module):
    def __init__(self, nKnapsackCategories, nThresholds, starting_thresholds, nineq=1, neq=0, eps=1e-8, cancel_rate_target=.05, cancel_rate_evaluation=.05, accept_rate_target=.75, accept_rate_evaluation=.75,
                 cancel_initializer=.02,inventory_initializer=3,cancel_coef_initializer=-.2,
                 cancel_intercept_initializer=.3,price_initializer=1,parametric_knapsack=False, knapsack_type=None):
        super().__init__()
        self.nKnapsackCategories = nKnapsackCategories
        self.nThresholds = nThresholds
        #self.nBatch = nBatch
        self.nineq = nineq
        self.neq = neq
        self.eps = eps
        self.cancel_rate_evaluation=cancel_rate_evaluation
        self.accept_rate_evaluation=accept_rate_evaluation
        self.benchmark_thresholds = Variable(starting_thresholds)
        #self.accept_rate_original=Parameter(accept_rate*torch.ones(1))
        #self.cancel_rate_original=Parameter(cancel_rate*torch.ones(1))
        #self.cancel_rate= self.cancel_rate_original*1.0
        #self.accept_rate=self.accept_rate_original*1.0
        self.accept_rate_param=Parameter(accept_rate_target*torch.ones(1))
        self.cancel_rate_param=Parameter(cancel_rate_target*torch.ones(1))
        self.inventory_initializer=inventory_initializer
        self.parametric_knapsack=parametric_knapsack
        self.h = Variable(torch.ones(self.nineq))
        ##Add matrix to make all variables >=0
        self.PosValMatrix = -1*Variable(torch.eye(self.nKnapsackCategories*self.nThresholds))
        self.PosValVector = Variable(torch.zeros(self.nKnapsackCategories*self.nThresholds))

        #Equality constraints. These will be the constraints to choose one variable per category
        ##These will be Variables as they are not something that is estimated by the model
        A = torch.zeros(self.nKnapsackCategories,self.nKnapsackCategories*self.nThresholds)
        for row in range(self.nKnapsackCategories):
            A[row][self.nThresholds*row:self.nThresholds*(row+1)]=1
        self.A=Variable(A)
        self.b = Variable(torch.ones(self.nKnapsackCategories))

        self.Q_zeros = Variable(torch.zeros(nKnapsackCategories*nThresholds,nKnapsackCategories*nThresholds))
        #Initialize thresholds
        self.thresholds = Variable(torch.arange(0,self.nThresholds))
        #Initialize cancel and revenue parameters
        if self.parametric_knapsack:
            self.thresholds_raw_matrix = Variable(starting_thresholds)
            #self.cancel_scale = Parameter((torch.rand(self.nKnapsackCategories)+.5)*cancel_initializer)
            #self.cancel_lam = Parameter(torch.ones(self.nKnapsackCategories)*cancel_initializer)
            #self.cancel_spread = Variable(torch.ones(self.nKnapsackCategories))
            #self.revenue_scale = Parameter((torch.rand(self.nKnapsackCategories)+.5))
            #self.revenue_lam = Parameter(torch.ones(self.nKnapsackCategories))
            #self.revenue_spread = Variable(torch.ones(self.nKnapsackCategories))
        else:
            #self.thresholds_raw_matrix = Parameter(torch.ones(self.nKnapsackCategories,self.nThresholds)*(1.0/self.nThresholds))
            self.thresholds_raw_matrix = Parameter(starting_thresholds)
        self.thresholds_raw_matrix_norm= torch.div(self.thresholds_raw_matrix,torch.sum(self.thresholds_raw_matrix,dim=1).expand_as(self.thresholds_raw_matrix))

        #Inventory distribution parameters
        self.inventory_lam_opt = Parameter(torch.ones(self.nKnapsackCategories)*inventory_initializer)
        #Cancel distribution parameters
        self.cancel_coef_opt = Parameter(torch.ones(self.nKnapsackCategories)*cancel_coef_initializer)
        self.cancel_intercept_opt = Parameter(torch.ones(self.nKnapsackCategories)*cancel_intercept_initializer)
        self.prices_opt = Parameter(torch.ones(self.nKnapsackCategories)*price_initializer)
        self.demand_distribution_opt = Parameter(torch.ones(self.nKnapsackCategories)*(1.0/self.nKnapsackCategories))

        self.inventory_lam_est = Parameter(torch.ones(self.nKnapsackCategories)*inventory_initializer)
        #Cancel distribution parameters
        self.cancel_coef_est = Parameter(torch.ones(self.nKnapsackCategories)*cancel_coef_initializer)
        self.cancel_intercept_est = Parameter(torch.ones(self.nKnapsackCategories)*cancel_intercept_initializer)
        self.prices_est = Parameter(torch.ones(self.nKnapsackCategories)*price_initializer)
        self.demand_distribution_est = Parameter(torch.ones(self.nKnapsackCategories)*(1.0/self.nKnapsackCategories))
    def normalize_thresholds(self):
        self.thresholds_raw_matrix.data.clamp_(min=self.eps,max=1.0-self.eps)
        param_sums = torch.ger(self.thresholds_raw_matrix.sum(dim=1).squeeze(),Variable(torch.ones(self.nThresholds)))
        self.thresholds_raw_matrix.data.div_(param_sums.data)
    def normalize_demand_params(self):
        self.demand_distribution_est.data.clamp_(min=self.eps)
        self.demand_distribution_est.data.div_(self.demand_distribution_est.data.sum())
        self.demand_distribution_opt.data.clamp_(min=self.eps)
        self.demand_distribution_opt.data.div_(self.demand_distribution_opt.data.sum())
    def forward(self,category,inv_count,price,cancel,collection_thresholds):
        #print("collection_thresholds",collection_thresholds)
        self.lp_infeasible=0
        self.cancel_coef_neg_est = self.cancel_coef_est.clamp(max=0)
        self.cancel_coef_neg_opt = self.cancel_coef_opt.clamp(max=0)
        self.nBatch = category.size(0)
        #x = x.view(nBatch, -1)

        #We want to compute everything we can without thresholds first. This will allow us to use our learned parameters to feed the LP
        self.inventory_distribution_raw_est = PoissonFunction(self.nKnapsackCategories,self.nThresholds,verbose=-1)(self.inventory_lam_est,self.thresholds)+self.eps
        self.inventory_distribution_norm_est = normalize_JK(self.inventory_distribution_raw_est,dim=1)
        self.inventory_distribution_batch_by_threshold_est = torch.mm(category,self.inventory_distribution_raw_est)+self.eps

        self.inventory_distribution_raw_opt = PoissonFunction(self.nKnapsackCategories,self.nThresholds,verbose=-1)(self.inventory_lam_opt,self.thresholds)+self.eps
        self.inventory_distribution_norm_opt = normalize_JK(self.inventory_distribution_raw_opt,dim=1)
        self.inventory_distribution_batch_by_threshold_opt = torch.mm(category,self.inventory_distribution_raw_opt)+self.eps


        ##Here we'll calculate cancel probability by inventory
        self.belief_cancel_rate_cXt_est = cancel_rate_belief_cXt(self.cancel_coef_neg_est,self.cancel_intercept_est,self.thresholds.unsqueeze(0).expand(self.nKnapsackCategories,self.nThresholds))
        belief_fill_rate_cXt_est = 1-self.belief_cancel_rate_cXt_est
        price_cXt_est = self.prices_est.unsqueeze(1).expand(self.nKnapsackCategories,self.nThresholds)

        ##Here we'll calculate cancel probability by inventory
        self.belief_cancel_rate_cXt_opt = cancel_rate_belief_cXt(self.cancel_coef_neg_opt,self.cancel_intercept_opt,self.thresholds.unsqueeze(0).expand(self.nKnapsackCategories,self.nThresholds))
        belief_fill_rate_cXt_opt = 1-self.belief_cancel_rate_cXt_opt
        price_cXt_opt = self.prices_opt.unsqueeze(1).expand(self.nKnapsackCategories,self.nThresholds)


        self.belief_total_demand_cXt_est = self.inventory_distribution_raw_est*(self.demand_distribution_est.unsqueeze(1).expand(self.nKnapsackCategories,self.nThresholds))
        belief_total_demand_c_vector_est = torch.sum(self.belief_total_demand_cXt_est,dim=1)

        self.belief_total_demand_cXt_opt = self.inventory_distribution_raw_opt*(self.demand_distribution_opt.unsqueeze(1).expand(self.nKnapsackCategories,self.nThresholds))
        belief_total_demand_c_vector_opt = torch.sum(self.belief_total_demand_cXt_opt,dim=1)

        if self.parametric_knapsack:

            self.belief_total_demand_opt = torch.sum(self.belief_total_demand_cXt_opt)
            self.belief_total_cancels_cXt_opt = self.belief_cancel_rate_cXt_opt*self.belief_total_demand_cXt_opt
            self.belief_total_fills_cXt_opt = belief_fill_rate_cXt_opt*self.belief_total_demand_cXt_opt
            self.knapsack_cancels_matrix = torch.div(torch.sum(self.belief_total_cancels_cXt_opt,dim=1).expand_as(self.belief_total_cancels_cXt_opt)-torch.cumsum(self.belief_total_cancels_cXt_opt,dim=1)+self.belief_total_cancels_cXt_opt,self.belief_total_demand_opt.expand(self.nKnapsackCategories,self.nThresholds))
            self.knapsack_fills_matrix = torch.div(torch.sum(self.belief_total_fills_cXt_opt,dim=1).expand_as(self.belief_total_fills_cXt_opt)-torch.cumsum(self.belief_total_fills_cXt_opt,dim=1)+self.belief_total_fills_cXt_opt,self.belief_total_demand_opt.expand(self.nKnapsackCategories,self.nThresholds))
            self.knapsack_revenues_matrix = self.knapsack_fills_matrix*price_cXt_opt
            self.knapsack_cancels = self.knapsack_cancels_matrix.view(1,-1)
            self.knapsack_fills = self.knapsack_fills_matrix.view(1,-1)
            self.knapsack_revenues = self.knapsack_revenues_matrix.view(-1)
            Q = self.Q_zeros + self.eps*Variable(torch.eye(self.nKnapsackCategories*self.nThresholds))
            self.inequalityMatrix = torch.cat((self.knapsack_cancels,-1*self.knapsack_fills,self.PosValMatrix))
            self.knapsack_cancels_RHS = torch.sum(self.knapsack_cancels_matrix*self.benchmark_thresholds)
            self.knapsack_fills_RHS = torch.sum(self.knapsack_fills_matrix*self.benchmark_thresholds)
            #self.inequalityVector = torch.cat((self.cancel_rate_param*self.h,-1*self.accept_rate_param*self.h,self.PosValVector))
            self.inequalityVector = torch.cat((self.knapsack_cancels_RHS*self.h,-1*self.knapsack_fills_RHS*self.h,self.PosValVector))
            try:
                thresholds_raw = QPFunctionJK(verbose=1)(Q, -1*self.knapsack_revenues, self.inequalityMatrix, self.inequalityVector, self.A, self.b)
                self.thresholds_raw_matrix = thresholds_raw.view(self.nKnapsackCategories,-1)
                #self.accept_rate=1.0*self.accept_rate_original
                #self.cancel_rate=1.0*self.cancel_rate_original
            except AssertionError:
                print("Error solving LP, likely infeasible")
                self.lp_infeasible=1
                #print("New Accept and Cancel Rates:",self.accept_rate,self.cancel_rate)
            self.thresholds_raw_matrix= F.relu(self.thresholds_raw_matrix)+self.eps
        self.thresholds_raw_matrix_norm = normalize_JK(self.thresholds_raw_matrix,dim=1)
        #This cXt matrix shows the probability of accepting an order under the learned thresholds, obtained either through direct optimization or through solving an LP
        accept_probability_cXt = torch.cumsum(self.thresholds_raw_matrix_norm,dim=1) #this gives the accept probability by cXt under parameterized thresholds


        #category is BxC matrix, so summing across dim 0 gets the number of accepted orders per category
        accept_probability_collection_bXt = torch.cumsum(collection_thresholds,dim=1)
        reject_probability_collection_bXt = 1-accept_probability_collection_bXt
        accept_percent_collection_bXt = accept_probability_collection_bXt*self.inventory_distribution_batch_by_threshold_est
        accept_percent_collection_b_vector = torch.sum(accept_percent_collection_bXt,dim=1).squeeze()#This is the believed acceptance rate of general orders of the categories corresponding with the batch under the collection thresholds
        reject_percent_collection_b_vector = 1-accept_percent_collection_b_vector
        self.batch_total_demand_b_vector = (1/accept_percent_collection_b_vector)#.clamp(min=0,max=100)


        #new to v37
        reject_percent_collection_expanded_bXt= reject_percent_collection_b_vector.unsqueeze(1).expand(self.nBatch,self.nThresholds)
        self.truncated_orders_distribution_bXt = torch.div(reject_probability_collection_bXt*self.inventory_distribution_batch_by_threshold_est,reject_percent_collection_expanded_bXt+self.eps)
        truncated_demand_b_vector = self.batch_total_demand_b_vector-1     #self.belief_total_demand_cXt
        truncated_demand_bXt = truncated_demand_b_vector.unsqueeze(1).expand(self.nBatch,self.nThresholds)*self.truncated_orders_distribution_bXt
        batch_total_demand_bXt = truncated_demand_bXt+inv_count
        self.batch_total_demand_cXt = torch.mm(category.t(),batch_total_demand_bXt)
        batch_total_demand_c_vector = torch.sum(self.batch_total_demand_cXt,dim=1)
        batch_zero_demand_c_vector = 1-batch_total_demand_c_vector.ge(0)
        #batch_supplement_demand = torch.masked_select(belief_total_demand_c_vector_est,batch_zero_demand_c_vector)
        self.estimated_batch_total_demand = torch.sum(self.batch_total_demand_b_vector)#+torch.sum(batch_supplement_demand)

        #Now we want to see how accurate our inventory distributions are for the batch
        accept_probability_batch_by_threshold = CumSumNoGrad(verbose=-1)(collection_thresholds)+self.eps
        self.inventory_distribution_batch_by_thresholds = torch.mm(category,self.inventory_distribution_raw_est)
        arrival_probability_batch_by_threshold_unnormed = self.inventory_distribution_batch_by_thresholds*accept_probability_batch_by_threshold
        arrival_probability_batch_by_threshold = torch.div(arrival_probability_batch_by_threshold_unnormed,torch.sum(arrival_probability_batch_by_threshold_unnormed,dim=1).expand_as(arrival_probability_batch_by_threshold_unnormed))
        log_arrival_prob = torch.log(arrival_probability_batch_by_threshold+self.eps)

        #Like we do for inventory, we want to measure the accuracy of our cancel params for the batch
        self.belief_cancel_rate_bXt = torch.mm(category, self.belief_cancel_rate_cXt_est)
        belief_fill_rate_bXt = 1-self.belief_cancel_rate_bXt
        self.belief_cancel_rate_b_vector = torch.sum(self.belief_cancel_rate_bXt*inv_count,dim=1).squeeze()
        belief_fill_rate_b_vector = 1-self.belief_cancel_rate_b_vector
        log_cancel_prob = torch.log(torch.cat((belief_fill_rate_b_vector.unsqueeze(1),self.belief_cancel_rate_b_vector.unsqueeze(1)),1)+self.eps)

        self.belief_category_dist_bXc = self.demand_distribution_est.unsqueeze(0).expand(self.nBatch,self.nKnapsackCategories)
        log_category_prob = torch.log(self.belief_category_dist_bXc+self.eps)

        ##This is new in v37. We want to combine the actual results observed in the batch but add in estimated effects of truncation

        accept_probability_using_threshold_params_bXt = torch.mm(category,accept_probability_cXt)
        truncated_accept_estimate = truncated_demand_bXt*accept_probability_using_threshold_params_bXt#This is the number of truncated orders we expect to accept (using param thresholds) at each inventory level corresponding to each order in the batch
        truncated_cancel_estimate = truncated_accept_estimate*self.belief_cancel_rate_bXt
        truncated_fill_estimate = truncated_accept_estimate*belief_fill_rate_bXt
        truncated_revenue_estimate = truncated_fill_estimate*(price.unsqueeze(1).expand(self.nBatch,self.nThresholds))
        truncated_revenue_estimate_sum = torch.sum(truncated_revenue_estimate)
        self.truncated_cancel_estimate_sum = torch.sum(truncated_cancel_estimate)
        truncated_fill_estimate_sum = torch.sum(truncated_fill_estimate)
        self.truncated_accept_estimate_sum = torch.sum(truncated_accept_estimate)

        ##This is new in v37. We want to combine the actual results observed in the batch but add in estimated effects of truncation
        fill = 1-cancel
        batch_cancel_bXt = cancel.unsqueeze(1).expand(self.nBatch,self.nThresholds)*inv_count*accept_probability_using_threshold_params_bXt
        batch_fill_bXt = fill.unsqueeze(1).expand(self.nBatch,self.nThresholds)*inv_count*accept_probability_using_threshold_params_bXt
        batch_cancel_b_vector = torch.sum(batch_cancel_bXt,dim=1).squeeze()
        batch_fill_b_vector = torch.sum(batch_fill_bXt,dim=1).squeeze()
        batch_accept_b_vector = torch.sum(inv_count*accept_probability_using_threshold_params_bXt,dim=1).squeeze()
        #print("sanity check",batch_accept_b_vector, batch_fill_b_vector+batch_cancel_b_vector)
        #print("sanity check 2", torch.sum(batch_accept_b_vector), torch.sum(batch_fill_b_vector+batch_cancel_b_vector))

        batch_revenue_b_vector = price*batch_fill_b_vector
        self.batch_fill_sum = torch.sum(batch_fill_b_vector,dim=0)
        self.batch_revenue_sum = torch.sum(batch_revenue_b_vector,dim=0)
        self.batch_cancel_sum = torch.sum(batch_cancel_b_vector,dim=0)
        self.batch_accept_sum = torch.sum(batch_accept_b_vector, dim=0)


        new_objective_loss = -(1.0/50000)*(truncated_revenue_estimate_sum+self.batch_revenue_sum)
        new_cancel_constraint_loss = self.truncated_cancel_estimate_sum+self.batch_cancel_sum-(self.truncated_accept_estimate_sum+self.batch_accept_sum)*self.cancel_rate_evaluation
        new_accept_constraint_loss = (1.0/7.0)*((self.truncated_accept_estimate_sum+self.batch_accept_sum)*self.accept_rate_evaluation-truncated_fill_estimate_sum-self.batch_fill_sum)
        #new_cancel_constraint_loss = truncated_cancel_estimate_sum+self.batch_cancel_sum-self.estimated_batch_total_demand*self.cancel_rate_param
        #new_accept_constraint_loss = (1.0/7.0)*(self.estimated_batch_total_demand*self.accept_rate_param-truncated_fill_estimate_sum-self.batch_fill_sum)

        observed_cancel_constraint_loss = self.batch_cancel_sum-(self.batch_accept_sum)*self.cancel_rate_evaluation
        observed_accept_constraint_loss = (1.0/7.0)*(self.batch_accept_sum*self.accept_rate_evaluation-self.batch_fill_sum)

        return new_objective_loss, new_cancel_constraint_loss, new_accept_constraint_loss, arrival_probability_batch_by_threshold, log_arrival_prob, log_cancel_prob, log_category_prob, self.estimated_batch_total_demand, observed_cancel_constraint_loss, observed_accept_constraint_loss, self.lp_infeasible