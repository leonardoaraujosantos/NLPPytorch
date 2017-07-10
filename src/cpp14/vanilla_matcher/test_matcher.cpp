#include "catch.hpp"
#include "conceptmatcher.h"

void PrintResp(const list<string> input){
    if (!input.empty()){
    std::copy( input.begin(),
                      input.end(),
                      std::ostream_iterator<std::string>(std::cout, "\n"));
    } else {
        cout << "NONE" << endl;
    }
}

TEST_CASE("Match concepts test"){
    // Catch test fixture
    // Create concept instance and add new concepts.
    ConceptMatcher concept = ConceptMatcher();
    concept.AddConcept(std::string("Indian"));
    concept.AddConcept(std::string("Thai"));
    concept.AddConcept(std::string("Sushi"));
    concept.AddConcept(std::string("Caribbean"));
    concept.AddConcept(std::string("Italian"));
    concept.AddConcept(std::string("West"));
    concept.AddConcept(std::string("East"));
    concept.AddConcept(std::string("Pub"));
    concept.AddConcept(std::string("Asian"));
    concept.AddConcept(std::string("BBQ"));
    concept.AddConcept(std::string("Chinese"));
    concept.AddConcept(std::string("Portuguese"));
    concept.AddConcept(std::string("Spanish"));
    concept.AddConcept(std::string("French"));
    concept.AddConcept(std::string("European"));

    SECTION("Question1"){
        std::string question = std::string("I would like some Thai food");
        cout << question << endl;
        list<string> matches = concept.GetMatches(question);
        PrintResp(matches);

        cout << "-----------END QUESTION-----------" << endl;
    }

    SECTION("Question2"){
        std::string question = std::string("Where can I find good sushi");
        cout << question << endl;
        list<string> matches = concept.GetMatches(question);
        PrintResp(matches);

        cout << "-----------END QUESTION-----------" << endl;
    }

    SECTION("Question3"){
        std::string question = std::string("Find me a place that does tapas");
        cout << question << endl;
        list<string> matches = concept.GetMatches(question);
        PrintResp(matches);

        cout << "-----------END QUESTION-----------" << endl;
    }

    SECTION("Question4"){
        std::string question = std::string("Which restaurants do East Asian food");
        cout << question << endl;
        list<string> matches = concept.GetMatches(question);
        PrintResp(matches);

        cout << "-----------END QUESTION-----------" << endl;
    }

    SECTION("Question5"){
        std::string question = std::string("Which restaurants do West Indian food");
        cout << question << endl;
        list<string> matches = concept.GetMatches(question);
        PrintResp(matches);

        cout << "-----------END QUESTION-----------" << endl;
    }

    SECTION("Question6"){
        std::string question = std::string("What is the weather like today");
        cout << question << endl;
        list<string> matches = concept.GetMatches(question);
        PrintResp(matches);

        cout << "-----------END QUESTION-----------" << endl;
    }

}
