import better_profanity
import random
import os
import torch
from IPython.display import Markdown



def _format_llm_output(text):
    """
    Function to apply formatting to the output from the LLMs.
    """
    return Markdown('<div class="alert alert-block alert-info">{}</div>'.format(text))


def _update_embeddings(model, tokenizer):
    
    # open file from code package that contains profanities
    with open(os.path.dirname(better_profanity.__file__)+'/profanity_wordlist.txt', 'r') as file:
        # read the file contents and store in list
        file_contents = file.read().splitlines()

    # get the current vocabulary
    vocabulary = tokenizer.get_vocab().keys()

    for word in file_contents:
        # check to see if new word is in the vocabulary or not
        if word not in vocabulary:
            tokenizer.add_tokens([word])

    # add new embeddings to the embedding matrix of the transformer model
    model.resize_token_embeddings(len(tokenizer))

    params = model.state_dict()
    
    # retrieve embeddings
    embeddings = params['encoder.embed_tokens.weight']
    
    # select original embeddings of model before resizing
    pre_expansion_embeddings = embeddings[:-len(file_contents),:]
    
    # calculate 
    mu = torch.mean(pre_expansion_embeddings, dim=0)
    n = pre_expansion_embeddings.size()[0]
    sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
    
    # update distribution
    dist = torch.distributions.multivariate_normal.MultivariateNormal(
            mu, covariance_matrix=1e-5*sigma)
    
    # create new embeddings with updated distribution
    new_embeddings = torch.stack(tuple((dist.sample() for _ in range(len(file_contents)))), dim=0)
    
    # assign new embeddings
    embeddings[-len(file_contents):,:] = new_embeddings
    
    # add new embeddings to state dict of model
    params['encoder.embed_tokens.weight'][-len(file_contents):,:] = new_embeddings
    
    # return
    return model, tokenizer