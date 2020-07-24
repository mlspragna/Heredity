import csv
import itertools
import sys
import copy

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    def cg(mg,fg):
        l=[mg,fg]
        if l==[2,2]:
            return [0.99*0.99,2*0.99*0.01,0.01*0.01]
        elif l==[0,0]:
            return [0.01*0.01,2*0.99*0.01,0.99*0.99]
        elif l==[2,0] or l==[0,2]:
            return [0.99*0.01,0.99*0.99+0.01*0.01,0.99*0.01]
        elif l==[2,1] or l==[1,2]:
            p1=cg(2,0)
            p2=cg(2,2)
            return [0.5*(p1[0]+p2[0]),0.5*(p1[1]+p2[1]),0.5*(p1[2]+p2[2])]
        elif l==[1,0] or l==[0,1]:
            p1=cg(0,0)
            p2=cg(2,0)
            return [0.5*(p1[0]+p2[0]),0.5*(p1[1]+p2[1]),0.5*(p1[2]+p2[2])]
        else:
            p1=cg(2,2)
            p2=cg(2,0)
            p3=cg(0,0)
            l1=[0,0,0]
            for i in range(0,3):
                l1[i]=0.25*p1[i]+0.5*p2[i]+0.25*p3[i]
            return l1
    p1=copy.deepcopy(people)
    for person in p1.values():
        person["pg"]=None
        person["pt"]=None
        if person["mother"]==None and person["father"]==None:
            if person["name"] in one_gene:
                person["pg"]=0.03
                if person["name"] in have_trait:
                    person["pt"]=PROBS["trait"][1][True]
                else:
                    person["pt"]=PROBS["trait"][1][False]
            elif person["name"] in two_genes:
                person["pg"]=0.01
                if person["name"] in have_trait:
                    person["pt"]=PROBS["trait"][2][True]
                else:
                    person["pt"]=PROBS["trait"][2][False]
            else:
                person["pg"]=0.96
                if person["name"] in have_trait:
                    person["pt"]=PROBS["trait"][0][True]
                else:
                    person["pt"]=PROBS["trait"][0][False]
    for person in p1.values():
        if person["mother"]!=None and person["father"]!=None:
            mg=0
            fg=0
            if person["mother"] in two_genes:
                mg=2
            elif person["mother"] in one_gene:
                mg=1
            else:
                mg=0
            if person["father"] in two_genes:
                fg=2
            elif person["father"] in one_gene:
                fg=1
            else:
                fg=0
            l=cg(mg,fg)
            if person["name"] in two_genes:
                person["pg"]=l[0]
                if person["name"] in have_trait:
                    person["pt"]=PROBS["trait"][2][True]
                else:
                    person["pt"]=PROBS["trait"][2][False]
            elif person["name"] in one_gene:
                person["pg"]=l[1]
                if person["name"] in have_trait:
                    person["pt"]=PROBS["trait"][1][True]
                else:
                    person["pt"]=PROBS["trait"][1][False]
            else:
                person["pg"]=l[2]
                if person["name"] in have_trait:
                    person["pt"]=PROBS["trait"][0][True]
                else:
                    person["pt"]=PROBS["trait"][0][False]
    joint_probability=1
    for person in p1.values():
        joint_probability=joint_probability*person["pt"]*person["pg"]
    return joint_probability
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    raise NotImplementedError


def update(probabilities, one_gene, two_genes, have_trait, p):
    for person in probabilities.keys():
        if person in two_genes:
            probabilities[person]["gene"][2]=probabilities[person]["gene"][2]+p
        elif person in one_gene:
            probabilities[person]["gene"][1]=probabilities[person]["gene"][1]+p
        else:
            probabilities[person]["gene"][0]=probabilities[person]["gene"][0]+p
        if person in have_trait:
            probabilities[person]["trait"][True]=probabilities[person]["trait"][True]+p
        else:
            probabilities[person]["trait"][False]=probabilities[person]["trait"][False]+p
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    raise NotImplementedError
    """

def normalize(probabilities):
    for person in probabilities.keys():
        x=probabilities[person]["gene"][2]+probabilities[person]["gene"][1]+probabilities[person]["gene"][0]
        probabilities[person]["gene"][2]=probabilities[person]["gene"][2]/x
        probabilities[person]["gene"][1]=probabilities[person]["gene"][1]/x
        probabilities[person]["gene"][0]=probabilities[person]["gene"][0]/x
        y=probabilities[person]["trait"][True]+probabilities[person]["trait"][False]
        probabilities[person]["trait"][True]=probabilities[person]["trait"][True]/y
        probabilities[person]["trait"][False]=probabilities[person]["trait"][False]/y
    return probabilities
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
