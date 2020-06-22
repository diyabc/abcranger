#pragma once

#include <catch2/catch.hpp>
#include <range/v3/all.hpp>

namespace Catch
{
    namespace Matchers
    {
        // namespace Vector {
        //     template<typename T>
        //     struct ApproxMatcher : MatcherBase<std::vector<T>> {

        //         ApproxMatcher(std::vector<T> const &comparator) : m_comparator( comparator ), approx{0.} {}

        //         bool match(std::vector<T> const &v) const override {
        //             if (m_comparator.size() != v.size())
        //                 return false;
        //             for (std::size_t i = 0; i < v.size(); ++i)
        //                 if (m_comparator[i] != approx(v[i]))
        //                     return false;
        //             return true;
        //         }
        //         std::string describe() const override {
        //             return "is approx: " + ::Catch::Detail::stringify( m_comparator );
        //         }
        //         template <typename = typename std::enable_if<std::is_constructible<double, T>::value>::type>
        //         ApproxMatcher& epsilon( T const& newEpsilon ) {
        //             approx.epsilon(newEpsilon);
        //             return *this;
        //         }
        //         template <typename = typename std::enable_if<std::is_constructible<double, T>::value>::type>
        //         ApproxMatcher& margin( T const& newMargin ) {
        //             approx.margin(newMargin);
        //             return *this;
        //         }
        //         template <typename = typename std::enable_if<std::is_constructible<double, T>::value>::type>
        //         ApproxMatcher& scale( T const& newScale ) {
        //             approx.scale(newScale);
        //             return *this;
        //         }

        //         std::vector<T> const& m_comparator;
        //         mutable ::Catch::Detail::Approx approx;
        //     };

        // }

        // template<typename T>
        // Vector::ApproxMatcher<T> Approx( std::vector<T> const& comparator ) {
        //     return Vector::ApproxMatcher<T>( comparator );

        // }

        namespace Range
        {
            template <typename Rng1, typename Rng2>
            struct ApproxMatcher : MatcherBase<Rng2>
            {
                typedef ranges::range_value_t<Rng1> T;

                ApproxMatcher(Rng1 const &comparator) : m_comparator(comparator), approx{0.} {}

                bool match(Rng2 const &v) const override
                {
                    if (ranges::distance(m_comparator) != ranges::distance(v))
                        return false;
                    auto ic = ranges::begin(m_comparator);
                    auto iv = ranges::begin(v);
                    for (; iv != ranges::end(v);)
                    {
                        if ((!std::isnan(*ic) || !std::isnan(*iv)) &&
                            (*ic != approx(*iv)))
                            return false;
                        ic = ranges::next(ic);
                        iv = ranges::next(iv);
                    }
                    return true;
                }

                std::string describe() const override
                {
                    return "is approx: " + ::Catch::Detail::stringify(m_comparator);
                }

                template <typename = typename std::enable_if<std::is_constructible<double, T>::value>::type>
                ApproxMatcher &epsilon(T const &newEpsilon)
                {
                    approx.epsilon(newEpsilon);
                    return *this;
                }
                template <typename = typename std::enable_if<std::is_constructible<double, T>::value>::type>
                ApproxMatcher &margin(T const &newMargin)
                {
                    approx.margin(newMargin);
                    return *this;
                }
                template <typename = typename std::enable_if<std::is_constructible<double, T>::value>::type>
                ApproxMatcher &scale(T const &newScale)
                {
                    approx.scale(newScale);
                    return *this;
                }

                Rng1 const &m_comparator;
                mutable ::Catch::Detail::Approx approx;
            };

            template <typename Rng1, typename Rng2>
            struct EqualsMatcher : MatcherBase<Rng2>
            {
                typedef ranges::range_value_t<Rng1> T;

                EqualsMatcher(const Rng1 &comparator) : m_comparator(comparator) {}

                std::string describe() const override
                {
                    return "is equals: " + ::Catch::Detail::stringify(m_comparator);
                }

                bool match(Rng2 const &v) const override
                {
                    if (ranges::distance(m_comparator) != ranges::distance(v))
                        return false;
                    auto ic = ranges::begin(m_comparator);
                    auto iv = ranges::begin(v);
                    for (; iv != ranges::end(v);)
                    {
                        if (*ic != *iv)
                            return false;
                        ic = ranges::next(ic);
                        iv = ranges::next(iv);
                    }
                    return true;
                }

                const Rng1 &m_comparator;
            };
        } // namespace Range

        template <typename Rng2,typename Rng1>
        Range::ApproxMatcher<Rng1, Rng2> ApproxRng(Rng1 const &comparator)
        {
            return Range::ApproxMatcher<Rng1, Rng2>(comparator);
        }

        template <typename Rng2, typename Rng1>
        Range::EqualsMatcher<Rng1, Rng2> EqualsRng(const Rng1 &comparator)
        {
            return Range::EqualsMatcher<Rng1, Rng2>(comparator);
        }

    } // namespace Matchers
} // namespace Catch