#ifndef LIST_HACK
#define LIST_HACK

class ListSet;

class ListHack {
public:
    ListHack(ListSet* ss) : sset(ss) {
    }

    long operator[](int idx);

    ListSet* sset;
};

#endif

